"""Flask application for creating and searching an ElasticSearch based Index."""

import hashlib
import os
from uuid import uuid4

from elasticsearch import Elasticsearch
from flask import Flask, jsonify, request

import squirro_chatbot.app_config as app_config
from squirro_chatbot.models.openai_chat_model import (
    OpenAIChatModel,
    OpenAIChatModelConfig,
)
from squirro_chatbot.search_result import Document, SearchResult

app = Flask(__name__)
app.config.from_object("app_config")


def _init_elasticsearch():
    """Initialize Python Elastic Search client."""

    es_username = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")

    # verify_certs set to False for this toy project.
    es = Elasticsearch(
        [app.config["ELASTICSEARCH_URL"]],
        http_auth=(es_username, es_password),
        use_ssl=True,
        verify_certs=False,
    )

    if not es.indices.exists(app.config["INDEX_NAME"]):
        es.indices.create(app.config["INDEX_NAME"])

    return es


# Initialize the elastic search client.
es = _init_elasticsearch()


def __init_openai_chatbot():

    openai_chat_config = OpenAIChatModelConfig(
        chat_model=app_config["chat_model"],
        max_tokens=app_config["max_tokens"],
        generation_length_tokens=app_config["generation_length_tokens"],
    )

    openai_chat_model = OpenAIChatModel(openai_chat_config)

    return openai_chat_model

openai_chat_model = __init_openai_chatbot()

@app.route("/documents/", methods=["POST"])
def create_document():
    """Creates and indexes a document.

    Returns:
        doc_id (str): Document ID of the given Document.
    """

    if not request.json or "text" not in request.json:
        return jsonify(error="Bad request"), 400

    document_text = request.json["text"]
    doc_id = hashlib.sha256(document_text.encode("utf-8")).hexdigest()

    es.index(index=app.config["INDEX_NAME"], id=doc_id, body={"text": document_text})

    return jsonify(document_id=doc_id)


@app.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    """Retrieves the document corresponding to the given doc id.

    Args:
        doc_id (str): Document ID

    Returns:
        text (str): Document text corresponding to the ID.
    """

    res = es.get(index=app.config["INDEX_NAME"], id=doc_id)

    if res["found"]:
        return jsonify(text=res["_source"]["text"])
    else:
        return jsonify(error="Document not found"), 404


@app.route("/search/", methods=["GET"])
def search_documents():
    """Returns the top k corresponding documents for a given query.

    Returns:
        results (List[Dict[str: str]]): List of relevant
            documents in the SearchResult format.
    """

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
