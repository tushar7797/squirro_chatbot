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
        chat_model=app.config["CHAT_MODEL"],
        max_tokens=app.config["MAX_TOKENS"],
        generation_length_tokens=app.config["GENERATION_LENGTH"]
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


def _retrieve_top_k_docs(query: str, top_k: int) -> str:
    """ """

    res = es.search(
        index=app.config["INDEX_NAME"],
        body={"query": {"match": {"text": query}}},
        size=top_k,
    )
    results = [
        SearchResult(
            document=Document(text=hit["_source"]["text"], id=hit["_id"]),
            score=hit["_score"],
        )
        for hit in res["hits"]["hits"]
    ]

    return results


@app.route("/search/", methods=["GET"])
def search_documents():
    """Returns the top k corresponding documents for a given query.

    Returns:
        results (List[Dict[str: str]]): List of relevant
            documents in the SearchResult format.
    """

    query = request.args.get("query", "")
    top_k = int(request.args.get("top_k", app.config["DEFAULT_TOP_K"]))
    results = _retrieve_top_k_docs(query=query, top_k=top_k)
    results = [result.model_dump() for result in results]

    return jsonify(results=results)


@app.route("/generate_answer/", methods=["GET"])
def generate_answer():

    query = request.args.get("query", "")
    top_k = app.config["DEFAULT_TOP_K"]

    relevant_docs = _retrieve_top_k_docs(query=query, top_k=top_k)
    prompt_context = " ".join([doc.document.text for doc in relevant_docs])

    prompt = f"Please Answer the query using the context provided. Query: {query}\nContext: {prompt_context}"
    answer = openai_chat_model.generate(prompt)

    return jsonify(answer = answer)

if __name__ == "__main__":
    app.run(debug=True)
