import hashlib
import os
from uuid import uuid4

import config
from elasticsearch import Elasticsearch
from flask import Flask, jsonify, request
from search_result import Document, SearchResult

app = Flask(__name__)
app.config.from_object("config")


def init_elasticsearch():
    es_username = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")

    es = Elasticsearch(
        app.config["ELASTICSEARCH_URL"],
        http_auth=(es_username, es_password),
        use_ssl=True,
        verify_certs=False,  # Reminder: Only set this to False for specific cases.
    )
    return es


es = init_elasticsearch()


@app.route("/documents/", methods=["POST"])
def create_document():

    if not request.json or "text" not in request.json:
        return jsonify(error="Bad request"), 400

    document_text = request.json["text"]
    doc_id = hashlib.sha256(document_text.encode("utf-8")).hexdigest()

    es.index(index=app.config["INDEX_NAME"], id=doc_id, body={"text": document_text})

    return jsonify(document_id=doc_id)


@app.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    res = es.get(index=app.config["INDEX_NAME"], id=doc_id)
    if res["found"]:
        return jsonify(text=res["_source"]["text"])
    else:
        return jsonify(error="Document not found"), 404


@app.route("/search/", methods=["GET"])
def search_documents():
    query = request.args.get("query", "")
    top_k = int(request.args.get("top_k", app.config["DEFAULT_TOP_K"]))
    res = es.search(
        index=app.config["INDEX_NAME"],
        body={"query": {"match": {"text": query}}},
        size=top_k,
    )
    results = [
        SearchResult(
            document=Document(text=hit["_source"]["text"], id=hit["_id"]),
            score=hit["_score"],
        ).model_dump()
        for hit in res["hits"]["hits"]
    ]

    return jsonify(results=results)


if __name__ == "__main__":
    app.run(debug=True)
