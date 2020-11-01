from db.ops import Document, Status, set_doc_properties_db, edit_db_doc
import add_text_layer
from utils import process_request
import os


def remove_ext(path):
    return os.path.splitext(path)[0]


def add_new_doc_db(doc):
    try:
        doc.process_status = Status.processing
        set_doc_properties_db(doc)
        path = doc.path
        path_wo_ext = remove_ext(doc.path)
        searchable_pdf_path = f"{path_wo_ext}_searchable.pdf"

        # make searchable pdf and txt
        add_text_layer.process_file(path, searchable_pdf_path)

        doc.searchable_pdf_path = searchable_pdf_path
        doc.txt_path = f"{remove_ext(searchable_pdf_path)}.txt"

        # classify vovik
        doc.doc_cls = process_request([doc.txt_path])[0]

        doc.process_status = Status.complete
 #     print("doc processed")
    except Exception as e:
     #   print(e)
        doc.process_status = Status.failed
    finally:
        edit_db_doc(doc.doc_id, doc.make_dict())


if __name__ == "__main__":
    doc = Document("228", path="/home/rickya/poedim/test/1 изм.pdf")
    add_new_doc_db(doc)
    print(doc.make_dict())
