# Squirro Chatbot

Squirro Chatbot is a powerful Python Flask-based application that leverages ElasticSearch and OpenAI's GPT-based models to provide an intelligent document retrieval and question-answering system. Users can submit documents, which are indexed and searchable, allowing the chatbot to retrieve relevant documents and generate answers to user queries based on the information contained within those documents.

## Features

- **Document Submission and Indexing:** Easily submit documents to be indexed using ElasticSearch, making them searchable for query answering.
- **Intelligent Query Answering:** Utilizes OpenAI's cutting-edge language models to generate accurate and relevant answers based on documents retrieved from the ElasticSearch index.
- **Flexible Document Retrieval:** Retrieve documents based on relevancy to the query, ensuring users get the most pertinent information.

## Getting Started

These instructions will guide you through the setup and deployment of Squirro Chatbot on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Flask
- Elasticsearch
- An OpenAI API key

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://yourrepository/squirro_chatbot.git
   cd squirro_chatbot
   ```

2. Install the repository in development mode

    ```sh
   pip install -e .
   ```
    We strongly suggest using a conda environment and then installing the repository.

3. Setup Environment Variables

    You will need to set environment variables for ElasticSearch and OpenAI API credentials:
    ```sh
    export ELASTICSEARCH_USERNAME='your_username'
    export ELASTICSEARCH_PASSWORD='your_password'
    export OPENAI_API_KEY='your_openai_api_key'
    export OPENAI_ORGANIZATION='your_openai_organization'
    ```

4. Running the Application

    ```sh
    python3 app.py
    ```
    
## Command to Create a document

curl -X POST http://127.0.0.1:5000/documents/ -H "Content-Type: application/json" -d '{"text": "This is a test document"}' 

## Command to Retrieve a document

curl http://127.0.0.1:5000/documents/c41cbbf2c21619e1d51dd729dbd9dd73693672ac0e358bfcda467827ba41bdf7

## Command to retrieve relevant documents

curl "http://127.0.0.1:5000/search/?query=test%20document&top_k=2"