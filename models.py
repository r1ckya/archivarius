import torch
from sru import SRU
from json import load
import re


class BBoxer(torch.nn.Module):
    def __init__(self, args):
        super(BBoxer, self).__init__()
        self.args = args
        self.mlp = torch.nn.Linear(10, 1)

    def forward(self, inp):
        x = self.mlp(inp)
        return x


class ImageExtracter(torch.nn.Module):
    def __init__(self, args):
        super(ImageExtracter, self).__init__()
        self.args = args
        self.mlp = torch.nn.Linear(10, 1)

    def forward(self, inp):
        x = self.mlp(inp)
        return x


class TextTokenizer(torch.nn.Module):
    def __init__(self, args):
        super(TextTokenizer, self).__init__()
        self.args = args
        self.create_model()
        with open(self.args["vocab_path"], "r") as f:
            self.vocab = load(f)
        self.vocab = {token: i + 1 for token, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def create_model(self):
        self.emb = torch.nn.Embedding(self.vocab_size, self.args["emb_dim"])

    def tokenize_texts(self, texts):
        min_token_len = self.args["min_token_len"]
        token_re = re.compile(r"[\w\d]+")
        return [
            [
                token
                for token in token_re.findall(text)
                if len(token) > min_token_len
            ]
            for text in texts
        ]

    def forward(self, texts):
        tokenized_text = self.tokenize_texts(texts)
        return tokenized_text


class ClassPredicter(torch.nn.Module):
    def __init__(self, args):
        super(ClassPredicter, self).__init__()
        self.args = args
        self.create_model()

    def create_model(self):
        if self.args["rnn_type"] == "conv":
            self.conv_layers = torch.nn.ModuleList()
            for i in range(self.args["rnn_conv_num_layers"]):
                self.conv_layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_channels=self.args["rnn_feat_dim"],
                            out_channels=self.args["rnn_feat_dim"],
                            kernel_size=self.args["rnn_kernel_size"],
                            padding=self.args["rnn_kernel_size"] // 2,
                        ),
                        torch.nn.Dropout(self.args["dropout"]),
                        torch.nn.LeakyReLU(),
                    )
                )

    def forward(self, flatten_images):
        """
        flatten_images = [(L_i*C), ..]
        """
        res = []
        if self.args["rnn_type"] == "conv":
            # TODO: process as batch
            for x in flatten_images:
                # 1*C*L
                x = torch.unsqueeze(x.permute(1, 0), 0)
                for layer in self.conv_layers:
                    x = x + layer(x)

                x = torch.squeeze(x.permute(1, 0), 0)
                # L*C
                x = self.mlp(x)
                x = torch.nn.Softmax(x)
                # L*V
                res.append(x)
        elif self.args["rnn_type"] == "sru":
            for x in flatten_images:
                x = self.rnn(x)
                x = self.mlp(x)
                x = torch.nn.Softmax(x)
                # L*V
                res.append(x)

        return res


class CharPredicter(torch.nn.Module):
    def __init__(self, args):
        super(CharPredicter, self).__init__()
        self.args = args
        self.create_model()

    def create_model(self):
        if self.args["rnn_type"] == "conv":
            self.conv_layers = torch.nn.ModuleList()
            for i in range(self.args["rnn_conv_num_layers"]):
                self.conv_layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_channels=self.args["rnn_feat_dim"],
                            out_channels=self.args["rnn_feat_dim"],
                            kernel_size=self.args["rnn_kernel_size"],
                            padding=self.args["rnn_kernel_size"] // 2,
                        ),
                        torch.nn.Dropout(self.args["dropout"]),
                        torch.nn.LeakyReLU(),
                    )
                )
        elif self.args["rnn_type"] == "sru":
            self.rnn = SRU(
                input_size=self.args["rnn_feat_dim"],
                hidden_size=self.args["rnn_feat_dim"],
                num_layers=self.args["rnn_num_stacked_layers"],
                dropout=self.args["dropout"],
                bidirectional=False,
                layer_norm=False,
                highway_bias=0,
                rescale=True,
            )

        self.mlp = torch.nn.Linear(
            self.args["rnn_feat_dim"], self.args["vocab_size"]
        )

    def forward(self, flatten_images):
        """
        flatten_images = [(L_i*C), ..]
        """
        res = []
        if self.args["rnn_type"] == "conv":
            # TODO: process as batch
            for x in flatten_images:
                # 1*C*L
                x = torch.unsqueeze(x.permute(1, 0), 0)
                for layer in self.conv_layers:
                    x = x + layer(x)

                x = torch.squeeze(x.permute(1, 0), 0)
                # L*C
                x = self.mlp(x)
                x = torch.nn.Softmax(x)
                # L*V
                res.append(x)
        elif self.args["rnn_type"] == "sru":
            for x in flatten_images:
                x = self.rnn(x)
                x = self.mlp(x)
                x = torch.nn.Softmax(x)
                # L*V
                res.append(x)

        return res


class TextPredicter(torch.nn.Module):
    def __init__(self, args):
        super(TextPredicter, self).__init__()
        self.args = args

    def forward(self, texts_probs):
        res = []
        for t in texts_probs:
            # vals, inds = torch.max(t, dim=1)
            # inds = torch.unique_consecutive(inds)
            # inds = inds[inds != 0]
            # need to del conseq

            # del blank symbol
            vals, inds = torch.max(t, dim=1)
            t = torch.cat([t, torch.unsqueeze(inds, 1)], 1)
            t = t[t[:, -1] == 0][:, -1]
            res.append(t)
        return res
