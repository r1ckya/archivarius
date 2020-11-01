import argparse
import os
from glob import glob
import pdf2image
from multiprocessing import Pool
import random


def merge_txt_files(file_list, out_file):
    with open(out_file, "w") as f:
        for fname in file_list:
            with open(fname) as infile:
                for line in infile:
                    f.write(line)


def file_list_to_string(file_list):
    cmd = ""
    for path in file_list:
        cmd = cmd + f'"{path}" '
    return cmd


def process_file(fin_path, fout_path):
    if (
        os.path.exists(fout_path)
        and os.path.exists(f"{fout_path[:-4]}.txt")
        and not os.stat(f"{fout_path[:-4]}.txt").st_size == 0
    ):
        return

    images = pdf2image.convert_from_path(fin_path, fmt="png")

    remove = []
    txt_files = []
    tmp_pdfs = []

    for i, img in enumerate(images):
        pref = f"{fout_path}.{i}"
        tmp_img = f"{pref}.png"
        tmp_pdf = f"{pref}.pdf"
        tmp_txt = f"{pref}.txt"
        # img = img.filter(ImageFilter.MedianFilter(size=3))
        img.save(tmp_img)

        cmd = (
            f'tesseract "{tmp_img}" "{tmp_pdf[:-4]}" --oem 1 -l rus+eng '
            f"pdf "
            f"txt "
        )  # f">/dev/null 2>&1"
        # print(cmd)
        os.system(cmd)

        txt_files.append(tmp_txt)
        tmp_pdfs.append(tmp_pdf)
        remove.append(tmp_pdf)
        remove.append(tmp_img)
        remove.append(tmp_txt)

    cmd = (
        f'gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE="{fout_path}" '
        f"-dBATCH {file_list_to_string(tmp_pdfs)}"
    )  # >/dev/null 2>&1"
    os.system(cmd)
    merge_txt_files(txt_files, f"{fout_path[:-4]}.txt")

    cmd = f"rm -f {file_list_to_string(remove)}"
    os.system(cmd)


def process_dir(din_path, dout_path):
    os.makedirs(dout_path, exist_ok=True)
    pdf_files = glob(os.path.join(din_path, "*.pdf"))
    args = []

    for fin_path in pdf_files:
        fout_path = os.path.join(dout_path, os.path.basename(fin_path))
        args.append((fin_path, fout_path))
        # process_file(*args[-1])
    return args


def process_dataset(dataset, n_jobs):
    dirs = glob(os.path.join(dataset, "*"))

    tasks = []
    for data_dir in dirs:
        if not data_dir.endswith("_processed"):
            tasks.extend(
                process_dir(data_dir, data_dir.strip("/") + "_processed")
            )

    random.shuffle(tasks)
    with Pool(n_jobs) as pool:
        pool.starmap(process_file, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset dir")
    parser.add_argument("n_jobs", type=int, help="", default=1)

    args = parser.parse_args()

    process_dataset(args.dataset, args.n_jobs)
