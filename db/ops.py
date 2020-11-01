from walrus import Database

from enum import Enum

import pymysql
from pymysql.cursors import DictCursor

host= "172.17.0.3" #"localhost"

def get_connection():
    return pymysql.connect(
        host=host,
        user="root",
        password="1234",
        port=3306,
        db="docs",
        charset="utf8mb4",
        cursorclass=DictCursor,
    )

def create_tables():
    connection = pymysql.connect(
        host=host,
        user="root",
        password="1234",
        port=3306,
        charset="utf8mb4",
        cursorclass=DictCursor,
    )
    with connection.cursor() as cursor:
        cursor.execute("CREATE DATABASE IF NOT EXISTS `docs`")
        connection.commit()
    connection.close()
    
    connection = get_connection()

    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS `docs`")
        cursor.execute("""CREATE TABLE docs(doc_id varchar(255), upload_id varchar(255), path varchar(255), process_status varchar(255), doc_cls varchar(255), searchable_pdf_path varchar(255), doc_src_name varchar(255), txt_path varchar(255), rawtext LONGTEXT, FULLTEXT (rawtext)) """)
        connection.commit()
    connection.close()


class Status(Enum):
    complete = 0
    processing = 1
    failed = 2


def add_document(key, content, **metadata):
    db = Database(host="localhost", port=6379, db=0)
    search_index = db.Index("app-search")
    search_index.add(key, content, **metadata)


def search_by_all_words(words):
    db = Database(host="localhost", port=6379, db=0)
    search_index = db.Index("app-search")
    return search_index.search(" AND ".join(words))


class Document:
    def __init__(self, doc_id, upload_id=None, path=None, process_status=None, doc_src_name=None):
        self.doc_id = doc_id
        self.upload_id = upload_id
        self.path = path
        self.process_status = process_status
        self.doc_src_name = doc_src_name

    def make_dict(self):
        res = {}
        if self.doc_id:
            res["doc_id"] = self.doc_id
        if self.upload_id:
            res["upload_id"] = self.upload_id
        if self.path:
            res["path"] = self.path
        if self.process_status:
            res["process_status"] = self.process_status.name.lower()
        if self.doc_src_name:
            res["doc_src_name"] = self.doc_src_name
        # classification
        if hasattr(self, "doc_cls"):
            res["doc_cls"] = self.doc_cls

        # text extraction
        if hasattr(self, "searchable_pdf_path"):
            res["searchable_pdf_path"] = self.searchable_pdf_path
        if hasattr(self, "txt_path"):
            res["txt_path"] = self.txt_path
            with open(self.txt_path, "r") as f:
                rawtext = f.read()
            res["rawtext"] = rawtext.replace('\n', ' ')

        return res


def get_doc_from_db(doc_id):
    connection = get_connection()
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `docs` WHERE `doc_id`=%s"
        cursor.execute(sql, (doc_id,))
        connection.commit()
        result = cursor.fetchone()
    connection.close()
    return result


def edit_db_doc(doc_id, edition):
    keys, values = zip(*edition.items())
    connection = get_connection()
    with connection.cursor() as cursor:
        set_args = ", ".join('`{}` = %s'.format(key) for key in keys)
        
        sql = "UPDATE `docs` SET {} WHERE `doc_id` = %s".format(set_args)

        args = tuple(list(values) + [doc_id])
        cursor.execute(sql, args)
        connection.commit()


def set_doc_properties_db(doc):
    connection = get_connection()
    keys, values = zip(*doc.make_dict().items())
    with connection.cursor() as cursor:
        sql = (
            "INSERT INTO `docs` ("
            + ", ".join("`" + key + "`" for key in keys)
            + ") VALUES ("
            + ", ".join("%s" for _ in values)
            + ")"
        )
        cursor.execute(sql, tuple(values))

    connection.commit()
    connection.close()


def search_docs(dict_value):
    connection = get_connection()
    text = None
    if 'text' in dict_value:
        text = dict_value.get('text')
        del dict_value['text']

    if len(dict_value):
        keys, values = zip(*dict_value.items())
    else:
        keys, values = [], []
    with connection.cursor() as cursor:
        exprs = ["(`{}`=%s)".format(key) for key in keys]
        if text: 
            exprs.append('MATCH (`rawtext`) AGAINST ("{}")'.format(text))
        
        where_expr = " AND ".join(exprs)
        sql = "SELECT * FROM `docs` WHERE " + where_expr
        cursor.execute(sql, tuple(values))
        connection.commit()
        result = cursor.fetchall()
    connection.close()
    return result


def get_process_status_by_upload_id(upload_id):
    connection = get_connection()
    with connection.cursor() as cursor:
        sql = "SELECT `process_status` FROM `docs` WHERE `upload_id`=%s"
        cursor.execute(sql, upload_id)
        connection.commit()
        ret = cursor.fetchall()

    statuses = set(x["process_status"] for x in ret)
    result = {}
    if Status.failed.name in statuses:
        status = Status.failed.name
    elif Status.processing.name in statuses:
        status = Status.processing.name
    elif Status.complete.name in statuses:
        status = Status.complete.name
        with connection.cursor() as cursor:
            sql = "SELECT * FROM `docs` WHERE `upload_id`=%s"
            cursor.execute(sql, upload_id)
            connection.commit()
            result = cursor.fetchall()
    else:
        status = "undefined"

    connection.close()
    return status, result


# s.decode("utf-8")
# bytes(s, "utf-8") # to bytes


if __name__ == "__main__":
    # # Phonetic search.
    # phonetic_index = db.Index('phonetic-search', metaphone=True)

    add_document("doc-1", "this is the content of document 1")
    add_document("doc-2", "another document", title="Another", status="1")
    add_document("doc-3", "ярик samal", title="hz", status="1")

    docs = search_by_all_words(["ярик", "yarik"])

    for document in docs:
        # Print the "title" that was stored as metadata. The "content" field
        # contains the original content of the document as it was indexed.
        print(document.get("title"), document["content"])
