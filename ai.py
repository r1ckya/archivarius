import os
from shutil import copytree, rmtree
from json import dump, load
from PIL import Image
#from torchvision import transforms
#from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
#from torch.utils.tensorboard import SummaryWriter
#from models import (
#    BBoxer,
#    ImageExtracter,
#    CharPredicter,
#    TextPredicter,
#    TextTokenizer,
#    ClassPredicter,
#)
import numpy as np
#import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class DocDataset:#(Dataset):
    def __init__(self, args):
        self.args = args
        self.path = args["tmp_data_dir"]
        self.clean_tmp_dir()
        self.prepare_data()

    def clean_tmp_dir(self):
        if os.path.exists(self.path):
            rmtree(self.path, ignore_errors=True)

    def split_files(self, files):
        import fitz

        counts = []

        for file in files:
            if not file.endswith(".pdf"):
                counts.append(1)
                continue
            doc = fitz.open(os.path.join(self.path, file))
            file = file[:-4]
            ct = 0
            for i in range(len(doc)):
                for img in doc.getPageImageList(i):
                    ct += 1
                    xref = img[0]
                    file_name = (
                        os.path.join(self.path, file) + "_" + str(ct) + ".png"
                    )
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5:  # this is GRAY or RGB
                        pix.writePNG(file_name)
                    else:  # CMYK: convert to RGB first
                        fitz.Pixmap(fitz.csRGB, pix)
                        pix.writePNG(file_name)
                    pix = None
            counts.append(ct)
            os.remove(os.path.join(self.path, file + ".pdf"))
        return counts

    def prepare_data(self):
        copytree(self.args["Dataset"], self.path)

        args = self.args
        folders = os.listdir(self.path)
        target_dict = dict()
        classmap = dict()
        files_list = []
        for folder_num, folder in enumerate(folders):
            classmap[folder_num] = folder
            cur_path = os.path.join(self.path, folder)
            cur_files = os.listdir(cur_path)
            for file in cur_files:
                file_path = os.path.join(cur_path, file)
                files_list.append(file_path)
                target_dict[file_path] = folder_num

        with open(args["classmap"], "w") as f:
            dump(classmap, f, indent=4)
        with open(args["target_file"], "w") as f:
            dump(target_dict, f, indent=4)

        self.classmap = classmap
        self.targets = target_dict
        self.file_list = files_list

    def do_transform(self, im):
        # TODO: denoise
        transform = transforms.Compose(
            [
                transforms.Resize(self.args["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(self.args["mean"], self.args["std"]),
            ]
        )
        return transform(im)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        im = Image.open(os.path.join(self.path, file))
        im = self.do_transform(im)
        return im, self.targets[file]


class SimpleDocDataset:
    def __init__(self, args):
        self.args = args
        self.path = args["tmp_data_dir"]
        self.clean_tmp_dir()
        self.prepare_data()

    def clean_tmp_dir(self):
        if os.path.exists(self.path):
            rmtree(self.path, ignore_errors=True)

    def prepare_data(self):
        copytree(self.args["Dataset"], self.path)

        args = self.args
        folders = os.listdir(self.path)
        target_dict = dict()
        classmap = dict()
        files_list = []
        for folder_num, folder in enumerate(folders):
            classmap[folder_num] = folder
            cur_path = os.path.join(self.path, folder)
            cur_files = os.listdir(cur_path)
            for file in cur_files:
                if not file.endswith(".txt"):
                    continue
                file_path = os.path.join(cur_path, file)
                files_list.append(file_path)
                target_dict[file_path] = folder_num

        with open(args["classmap"], "w") as f:
            dump(classmap, f, indent=4)
        with open(args["target_file"], "w") as f:
            dump(target_dict, f, indent=4)

        self.classmap = classmap
        self.targets = target_dict
        self.file_list = files_list

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        with open(file, "r") as f:
            text = f.read()
        return text, self.targets[file]

    def simple_search(self):
        with open(self.args["semanticwords"], "r") as f:
            target2tokens = load(f)
        class2id = {self.classmap[k]: k for k in self.classmap}
        predictions = []
        for text, target in self:
            scores = np.zeros(len(target2tokens))
            text = text.lower()
            for t in target2tokens:
                tokens = target2tokens[t]
                for tartoken in tokens:
                    scores[class2id[t]] += text.count(tartoken)
            predictions.append(np.argmax(scores))

        rep = classification_report(
            list(self.targets.values()),
            predictions,
            target_names=list(self.classmap.values()),
        )

        matrix = confusion_matrix(
            y_true=list(self.targets.values()),
            y_pred=predictions,
        )

        df_cm = pd.DataFrame(
            matrix,
            index=[list(self.classmap.values())],
            columns=[list(self.classmap.values())],
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig("confusion_matrix")
        print(rep)
        return None


class AI(object):
    """docstring for AI"""

    def __init__(self, args):
        super(AI, self).__init__()
        self.args = args

    def process(self, files):
        mode = self.args["mode"]
        if mode == "train":
            self.dataset = DocDataset(self.args)
        elif mode == "inference" and files is None:
            self.dataset = SimpleDocDataset(self.args)

        if mode == "train":
            resp = self.train()
        elif mode == "inference":
            resp = self.inference(files)
        return resp

    def simple_search(self, files):
        with open(self.args["semanticwords"], "r") as f:
            target2tokens = load(f)
        with open(self.args["classmap"], "r") as f:
            classmap = load(f)
        classmap = {int(k): classmap[k] for k in classmap}
        class2id = {classmap[k]: k for k in classmap}
        predictions = []
        for file in files:
            with open(file, "r") as f:
                text = f.read()
            scores = np.zeros(len(target2tokens))
            text = text.lower()
            for t in target2tokens:
                tokens = target2tokens[t]
                for tartoken in tokens:
                    scores[class2id[t]] += text.count(tartoken)
            predictions.append(np.argmax(scores))

        return [classmap[p] for p in predictions]

    def create_model(self):
        task = self.args["task_type"]
        if task == "image_extract":
            self.model = torch.nn.Sequential(
                BBoxer(self.args), ImageExtracter(self.args)
            )
        elif task == "classify_text":
            self.model = torch.nn.Sequential(
                TextTokenizer(self.args), ClassPredicter(self.args)
            )
        elif task == "generate_text":
            self.model = torch.nn.Sequential(
                BBoxer(self.args),
                ImageExtracter(self.args),
                CharPredicter(self.args),
                TextPredicter(self.args),
            )
        self.model.to(self.device)

    def gen_loaders(self):
        batch_size = self.args["batch_size"]
        indices = list(range(len(self.dataset)))
        np.random.shuffle(indices)
        split = self.args["split"]
        size_train = int(len(self.dataset) * split[0])
        size_val = int(len(self.dataset) * split[1])
        train_indices, val_indices, test_indices = (
            indices[size_train:],
            indices[size_train : size_train + size_val],
            indices[size_train + size_val :],
        )
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        dataloader_train = DataLoader(
            dataset=self.dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
            sampler=train_sampler,
        )
        dataloader_val = DataLoader(
            dataset=self.dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
            sampler=val_sampler,
        )
        dataloader_test = DataLoader(
            dataset=self.dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
            sampler=test_sampler,
        )
        return dataloader_train, dataloader_val, dataloader_test

    def write_metrics(self, writer, Y, inference, phase, it):
        if (
            self.args["task"] == "classify_text"
            or self.args["task"] == "generate_text"
        ):
            target = Y.detach().cpu().numpy()
            source = inference.detach().cpu().numpy()
            acc = accuracy_score(target, source)
            f1 = f1_score(target, source)
            auc = roc_auc_score(target, source)
            if phase == "test":
                writer.add_hparams(
                    {
                        "lr": self.args["lr"],
                        "bsize": self.args["bsize"],
                        "epochs": self.args["epochs"],
                        "image_size": self.args["image_size"],
                        "mean": self.args["mean"],
                        "std": self.args["std"],
                        "split": self.args["split"],
                    },
                    {"hparam/acc": acc, "hparam/f1": f1, "hparam/auc": auc},
                )
            else:
                writer.add_scalar(
                    "metrics/acc_" + phase,
                    acc,
                    global_step=it,
                )
                writer.add_scalar(
                    "metrics/f1_" + phase,
                    f1,
                    global_step=it,
                )
                writer.add_scalar(
                    "metrics/auc_" + phase,
                    auc,
                    global_step=it,
                )
            return acc, f1, auc

    def train(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.create_model()
        task = self.args["task_type"]
        if task == "image_extract":
            loss_fn = torch.nn.MSELoss()
        elif task == "classify_text" or task == "generate_text":
            loss_fn = torch.nn.CrossEntropyLoss()

        dataloader_train, dataloader_val, dataloader_test = self.gen_loaders()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args["lr"], weight_decay=1e-5
        )
        writer = SummaryWriter(self.args["tb_path"])
        best_auc = 0
        for epoch in range(self.args["epochs"]):
            self.model.train()
            for i, (X, Y) in enumerate(dataloader_train):
                X, Y = X.to(self.device), Y.to(self.device)
                optimizer.zero_grad()
                inference = self.model.forward(X)
                loss = loss_fn(inference, Y)
                self.write_metrics(
                    writer,
                    Y,
                    inference,
                    "train",
                    i + self.args["batch_size"] * epoch,
                )
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.model.eval()
                for i, (X, Y) in enumerate(dataloader_val):
                    X, Y = X.to(self.device), Y.to(self.device)
                    inference = self.model.forward(X)
                    loss = loss_fn(inference, Y)
                    metrics = self.write_metrics(
                        writer,
                        Y,
                        inference,
                        "val",
                        i + self.args["batch_size"] * epoch,
                    )
                    if metrics is not None:
                        acc, f1, auc = metrics
                    if auc > best_auc:
                        best_auc = auc
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.args["tb_path"], "weights.ckpt"),
                        )

        with torch.no_grad():
            self.model.eval()
            for i, (X, Y) in enumerate(dataloader_test):
                X, Y = X.to(self.device), Y.to(self.device)
                inference = self.model.forward(X)
                loss = loss_fn(inference, Y)
                self.write_metrics(writer, Y, inference, "test")
        writer.close()

    def inference(self, files):
        if self.args["ai_type"] == "simple_search":
            assert files is not None, files
            if files is not None:
                return self.simple_search(files)
            else:
                return self.dataset.simple_search()
