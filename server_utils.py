import random
import string
import os.path
import zipfile


preset = {
    "pdf_library_path": "/var/www/uploads/pdf/",
    "zip_library_path": "/var/www/uploads/zip/",
    "len_name": 10,
}

ARCHIVE_EXTENSIONS = set(["zip"])
FILE_EXTENSIONS = set(["pdf"])
ALLOWED_EXTENSIONS = ARCHIVE_EXTENSIONS | FILE_EXTENSIONS


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1] in ALLOWED_EXTENSIONS


def is_archive(filename):
    return "." in filename and filename.rsplit(".", 1)[-1] in ARCHIVE_EXTENSIONS


def is_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1] in FILE_EXTENSIONS


def construct_filename_by_doc_id(doc_id):
    return preset["pdf_library_path"] + doc_id + ".pdf"


def construct_zip_path_by_upload_id(upload_id):
    return preset["zip_library_path"] + upload_id + ".zip"


def generate_random_string(len_name=preset["len_name"]):
    return "".join(random.choice(string.digits) for _ in range(len_name))


def generate_pdf_path():
    doc_id = generate_random_string()
    file_path = construct_filename_by_doc_id(doc_id)
    return file_path, doc_id


def generate_zip_path():
    upload_id = generate_random_string()
    file_path = construct_zip_path_by_upload_id(upload_id)
    return file_path, upload_id


def generate_upload_id():
    pass


def generate_doc_id():
    file_path, doc_id = generate_pdf_path()
    while os.path.exists(file_path):
        file_path, doc_id = generate_pdf_path()
    return file_path, doc_id


def generate_edit_id():
    pass


def extract_all(zip_path):
    dst_path = zip_path[:-4] + "/"

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dst_path)

    return dst_path
