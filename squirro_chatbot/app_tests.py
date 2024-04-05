"""Tests for functionalities in the main app."""

from absl.testing import absltest
from unittest.mock import patch, MagicMock
import json
from flask import Flask
from squirro_chatbot.app import app
from squirro_chatbot.search_result import SearchResult, Document
from squirro_chatbot.app import _retrieve_top_k_docs


class TestFlaskApp(absltest.TestCase):

    def setUp(self):
        """Set up a test client for the Flask app."""
        self.app = app.test_client()
        self.app.testing = True

    @patch("squirro_chatbot.app.es")
    @patch("squirro_chatbot.app.hashlib.sha256")
    def test_create_document(self, mock_sha256, mock_es):
        """Test creating and indexing a new document with a mocked document ID."""
        # Mock the hexdigest method to return '1' instead of a real hash.
        mock_sha256.return_value.hexdigest.return_value = "1"

        mock_es.index.return_value = {}
        response = self.app.post("/documents/", json={"text": "Test document text"})

        expected_response = {"document_id": "1"}
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), expected_response)

    @patch("squirro_chatbot.app.es")
    def test_get_document_found(self, mock_es):
        """Test retrieving a document that exists."""
        mock_es.get.return_value = {"found": True, "_source": {"text": "Document text"}}
        response = self.app.get("/documents/some_document_id")
        self.assertEqual(response.status_code, 200)
        expected_response = {"text": "Document text"}
        self.assertEqual(json.loads(response.data), expected_response)

    @patch("squirro_chatbot.app.es")
    def test_get_document_not_found(self, mock_es):
        """Test retrieving a document that does not exist."""
        mock_es.get.return_value = {"found": False}
        response = self.app.get("/documents/nonexistent_document_id")
        expected_response = {"error": "Document not found"}
        self.assertEqual(response.status_code, 404)
        self.assertEqual(json.loads(response.data), expected_response)

    @patch("squirro_chatbot.app.es")
    @patch("squirro_chatbot.app._retrieve_top_k_docs")
    def test_search_documents(self, mock_retrieve_top_k_docs, mock_es):
        """Test the search functionality."""
        mock_retrieve_top_k_docs.return_value = [
            SearchResult(document=Document(text="Document 1 text", id="1"), score=0.9),
            SearchResult(document=Document(text="Document 2 text", id="2"), score=0.8),
            SearchResult(document=Document(text="Document 3 text", id="3"), score=0.7),
        ]
        response = self.app.get("/search/?query=test&top_k=3")
        expected_response = {
            "results": [
                {"document": {"id": "1", "text": "Document 1 text"}, "score": 0.9},
                {"document": {"id": "2", "text": "Document 2 text"}, "score": 0.8},
                {"document": {"id": "3", "text": "Document 3 text"}, "score": 0.7},
            ]
        }
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), expected_response)

    @patch("squirro_chatbot.app.openai_chat_model")
    @patch("squirro_chatbot.app._retrieve_top_k_docs")
    def test_generate_answer(self, mock_retrieve_top_k_docs, mock_openai_chat_model):
        """Test generating an answer using the OpenAI Chat model."""
        mock_retrieve_top_k_docs.return_value = [
            SearchResult(document=Document(text="Document 1 text", id="1"), score=0.9),
            SearchResult(document=Document(text="Document 2 text", id="2"), score=0.8),
            SearchResult(document=Document(text="Document 3 text", id="3"), score=0.7),
        ]
        mock_openai_chat_model.generate.return_value = "Generated answer"
        response = self.app.get("/generate_answer/?query=test")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Generated answer")

    @patch("squirro_chatbot.app.es")
    def test_retrieve_top_k_docs(self, mock_es):
        """Test the _retrieve_top_k_docs functionality."""
        # Mock Elasticsearch search response
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {"text": "First document text", "id": "1"},
                        "_score": 0.9,
                        "_id": "1",
                    },
                    {
                        "_source": {"text": "Second document text", "id": "2"},
                        "_score": 0.8,
                        "_id": "2",
                    },
                    {
                        "_source": {"text": "Third document text", "id": "3"},
                        "_score": 0.7,
                        "_id": "3",
                    },
                ]
            }
        }

        expected_results = [
            SearchResult(
                document=Document(text="First document text", id="1"), score=0.9
            ),
            SearchResult(
                document=Document(text="Second document text", id="2"), score=0.8
            ),
            SearchResult(
                document=Document(text="Third document text", id="3"), score=0.7
            ),
        ]
        # Execute the function under test
        results = _retrieve_top_k_docs("test query", 3)
        self.assertEqual(results, expected_results)


if __name__ == "__main__":
    absltest.main()
