from flask import Flask, request, send_file, send_from_directory, make_response
import json
import db.ops as ops
import os
import xlsxwriter
import shutil
import codecs
from multiprocessing import Process
import server_utils
import pymysql
import classify_doc
import argparse

try:
    from werkzeug import secure_filename
except ImportError:
    from werkzeug.utils import secure_filename

STATIC_PATH = "build"
app = Flask(__name__, static_url_path='', static_folder=STATIC_PATH)


@app.route('/')
def hello():
    return app.send_static_file('index.html')

@app.route('/<name>/<id>')
def front_path1(name, id):
    return app.send_static_file('index.html')

@app.route('/<name>')
def front_path(name):
    return app.send_static_file('index.html')



def process_pdf_wrapper(doc):
    #return doc
    p = Process(target=classify_doc.add_new_doc_db, args=(doc,))
    p.daemon = True
    p.start()


@app.route("/api/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        user_filename = secure_filename(f.filename)

        
        if server_utils.is_archive(user_filename):

            file_path, upload_id = server_utils.generate_zip_path()
            f.save(file_path)
            extracted_files_path = server_utils.extract_all(file_path)

            for doc_src_path in os.listdir(extracted_files_path):
                doc_src_path = os.path.join(extracted_files_path, doc_src_path)
                if not os.path.isfile(doc_src_path):
                    continue
                doc_dst_path, doc_id = server_utils.generate_pdf_path()
                shutil.copy(doc_src_path, doc_dst_path)
                #f.save(doc_dst_path)
                doc = ops.Document(
                    doc_id, path=doc_dst_path, upload_id=upload_id, doc_src_name=doc_src_path.rsplit('/')[-1]
                )
                # ops.set_doc_properties_db(doc)
                process_pdf_wrapper(doc)
            return json.dumps({"result": "ok", "upload_id": upload_id})

        elif server_utils.is_file(user_filename):
            file_path, doc_id = server_utils.generate_pdf_path()
            upload_id = doc_id
            f.save(file_path)
            doc = ops.Document(doc_id, path=file_path, upload_id=upload_id, doc_src_name=f.filename)
            # ops.set_doc_properties_db(doc)
            process_pdf_wrapper(doc)
            return json.dumps({"result": "ok", "upload_id": upload_id})
        else:
            return "broken"
    return "whatever"


@app.route("/api/upload/<upload_id>", methods=["GET", "POST"])
def upload_status(upload_id):
    status, parsed_list = ops.get_process_status_by_upload_id(upload_id)
    result = {"process_status": status}
    if status == ops.Status.complete.name:
        result["result"] = parsed_list
    return json.dumps(result)


# return xlsx here
# @app.route("/api/upload/<upload_id>/download")
# def upload_file_xlsx(upload_id):
#     pass


@app.route("/api/edit/<doc_id>", methods=["GET", "POST"])
def edit_doc(doc_id):
    if request.method == "POST":
        edition = request.json
        # doc = ops.get_doc_from_db(doc_id)
        ops.edit_db_doc(doc_id, edition)
        return json.dumps({"result": "ok"})
    return "whatever"


@app.route("/api/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query = request.json
        resp = ops.search_docs(query)
        return json.dumps(resp)
    return "whatever"


# @app.route("/api/search/download")
# def search_xlsx():
#     pass


@app.route("/api/view/<doc_id>", methods=["GET", "POST"])
def view_doc(doc_id):
    result = ops.get_doc_from_db(doc_id)
    return json.dumps(result)


@app.route("/api/pdf/<doc_id>", methods=["GET", "POST"])
def send_doc(doc_id):
    return send_from_directory(
        directory=server_utils.preset["pdf_library_path"],
        filename=ops.get_doc_from_db(doc_id)["searchable_pdf_path"].rsplit('/')[-1],
    )

@app.route("/api/info/classes", methods=["GET", "POST"])
def info_classes():
    with open("data/classmap.json", "r") as f:
        s = f.read()
    return s


@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', dest="init", action="store_true")

    args = parser.parse_args()
    if args.init:
        print("reinit database..")
        ops.create_tables()
    # app.run()
    app.run(host="0.0.0.0", port=5000)
