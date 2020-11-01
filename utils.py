import ai
import os


def process_request(files):
    """
    args:
        image_size = (int1, int2)
        mean = [m1, m2, m3]
        std = [s1, s2, s3]
        batch_size = int
        lr = float
        epochs = int
        tb_path = str
        split = [p1, p2, p3], p1 + p2 + p3 == 1
        rnn_conv_num_layers = int
        dropout = float
        rnn_type = 'conv' or 'gru' or 'lstm' or 'sru'
        rnn_feat_dim = int
        rnn_kernel_size = int
        vocab_path = str
        rnn_num_stacked_layers = int
        emb_dim = int
        min_token_len = int
        target_file = {f1: str1, f2: str2...}
        ai_type = 'simple_search' or 'rnn'
        tmp_data_dir = str
        mode = 'inference' or 'train'
        task_type = 'image_extract' or 'classify_text' or 'generate_text'
        Dataset = str
        classmap = str
        semanticwords = str
    """
    args = {
        "image_size": (600, 600),
        "mean": [0.5, 0.5, 0.5],
        "std": [1, 1, 1],
        "batch_size": 2,
        "lr": 0.001,
        "epochs": 2,
        "tb_path": "runs/1",
        "split": [0.6, 0.2, 0.2],
        "rnn_conv_num_layers": 3,
        "dropout": 0.4,
        "rnn_feat_dim": 16,
        "rnn_kernel_size": 3,
        "vocab_size": 72,
        "rnn_num_stacked_layers": 2,
        "emb_dim": 16,
        "min_token_len": 4,
        "target_file": "data/target.json",
        "ai_type": "simple_search",
        "tmp_data_dir": "tmp",
        "mode": "inference",
        "task_type": "classify_text",
        "Dataset": "Dataset2",
        "classmap": "data/classmap.json",
        "semanticwords": "mostsemanticword.json",
    }
    AI = ai.AI(args)
    response = AI.process(files)
    return response


if __name__ == "__main__":
    files = [
        "/home/vladimir/programming/hackaton/tmp/БТИ_processed/1. ул Шеногина, дом 3, строение 14 изм.txt",
        "/home/vladimir/programming/hackaton/tmp/БТИ_processed/9. Летниковская улица, дом 11-11, строение 7 изм.txt",
        "/home/vladimir/programming/hackaton/tmp/ЗУ_processed/M-03-040506 исправленное.txt",
        "/home/vladimir/programming/hackaton/tmp/Свид. АГР_processed/54.txt",
    ]
    process_request(files)
