"""Flask application for creating and searching an ElasticSearch based Index.

The App has the following functions/endpoints:

    create_document: Creates a document and returns the document id.
    get_document: Retrieves the document given the document id.
    search_documents: Retrieves the top k documents for a given query.
    generate_answer: Generates an answer given the search query by 
        retrieving the most relevant documents and using OpenAIChatModel.
"""

import hashlib
import os
from uuid import uuid4

from elasticsearch import Elasticsearch
from flask import Flask, jsonify, request
from flask_cors import CORS

import squirro_chatbot.app_config as app_config
from squirro_chatbot.models.openai_chat_model import (
    OpenAIChatModel,
    OpenAIChatModelConfig,
)
from squirro_chatbot.search_result import Document, SearchResult

# Initialize Flask app and apply configurations
app = Flask(__name__)
app.config.from_object("app_config")
CORS(app)

# Both are initialized with initialize_app
elastic_search_client = None
openai_chat_model = None


def initialize_app():
    """Initialises all the essentials for the app.

    Initializes the elastic search client and openai_chat_model
    according to provided configs.
    """
    global elastic_search_client
    global openai_chat_model

    # Initialize elastic search client.
    es_username = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")

    # verify_certs set to False for this toy project.
    elastic_search_client = Elasticsearch(
        [app.config["ELASTICSEARCH_URL"]],
        http_auth=(es_username, es_password),
        use_ssl=True,
        verify_certs=False,
    )

    if not elastic_search_client.indices.exists(app.config["INDEX_NAME"]):
        elastic_search_client.indices.create(app.config["INDEX_NAME"])

    # Initialize the Open AI Chat Model.
    openai_chat_config = OpenAIChatModelConfig(
        chat_model=app.config["CHAT_MODEL"],
        max_tokens=app.config["MAX_TOKENS"],
        generation_length_tokens=app.config["GENERATION_LENGTH"],
    )

    openai_chat_model = OpenAIChatModel(openai_chat_config)


@app.route("/documents/", methods=["POST"])
def create_document():
    """Endpoint for creating and indexing a new document.

    Expects a JSON payload with a "text" field. Document indexed
    in Elastic Search Index with id set to SHA-256 hash of the text

    Returns
        A JSON response containing the document_id of the indexed document.
    """

    if not request.json or "text" not in request.json:
        return jsonify(error="Bad request"), 400

    document_text = request.json["text"]
    doc_id = hashlib.sha256(document_text.encode("utf-8")).hexdigest()

    elastic_search_client.index(
        index=app.config["INDEX_NAME"], id=doc_id, body={"text": document_text}
    )

    return jsonify(document_id=doc_id)


@app.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id: str):
    """Retrieves the document corresponding to the given doc id.

    Args:
        doc_id: Document ID

    Returns:
        A JSON response containing the `text` of the document if found,
        otherwise an error message and a 404 status code.
    """

    res = elastic_search_client.get(index=app.config["INDEX_NAME"], id=doc_id)

    if res["found"]:
        return jsonify(text=res["_source"]["text"])
    else:
        return jsonify(error="Document not found"), 404


def _retrieve_top_k_docs(query: str, top_k: int) -> list[SearchResult]:
    """Retrieves the top k relevant documents for query.

    Args:
        query: Query to be searched
        top_k: Number of documents to be retrieved

    Returns:
        results: List of Most relevant docs.
    """

    # Match the document with the query
    res = elastic_search_client.search(
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
    """Retrieves top k documents matching a query.

    Args:
        query: The search query to match documents.
        top_k: The number of top matching documents to retrieve.

    Returns:
        JSON Response of list of SearchResult objects representing
          the top k matching documents.
    """

    query = request.args.get("query", "")
    top_k = int(request.args.get("top_k", app.config["DEFAULT_TOP_K"]))
    results = _retrieve_top_k_docs(query=query, top_k=top_k)
    results = [result.model_dump() for result in results]

    return jsonify(results=results)


@app.route("/generate_answer/", methods=["GET"])
def generate_answer():
    """Generate Open AI Chat model response to query.

    Returns:
        JSON Response with generated answer and retrieved
            document ids.
    """
    query = request.args.get("query", "")
    top_k = app.config["DEFAULT_TOP_K"]

    relevant_docs = _retrieve_top_k_docs(query=query, top_k=top_k)
    prompt_context = " ".join([doc.document.text for doc in relevant_docs])

    prompt = f"Please Answer the query using the context provided. Query: {query}\nContext: {prompt_context}"
    answer = openai_chat_model.generate(prompt)

    retrieved_doc_ids = [result.document.id for result in relevant_docs]

    return jsonify(answer=answer, retrieved_doc_ids=retrieved_doc_ids)


if __name__ == "__main__":
    initialize_app()
    app.run(debug=True)
