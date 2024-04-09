# Squirro Chatbot

[Detailed Documentation Link](https://docs.google.com/document/d/1Q5KtNZrrqpJIt-NZx7JRqYFKYO027fVnVuEMmTXUa-4/edit?usp=sharing)

Squirro Chatbot is a Python Flask-based application that leverages ElasticSearch and OpenAI's GPT-based models to provide an intelligent document retrieval and question-answering system. Users can submit documents, which are indexed and searchable, allowing the chatbot to retrieve relevant documents and generate answers to user queries based on the information contained within those documents.

## Features

- **Document Submission and Indexing:** Easily submit documents to be indexed using ElasticSearch, making them searchable for query answering.
- **Intelligent Query Answering:** Utilizes OpenAI's cutting-edge language models to generate accurate and relevant answers based on documents retrieved from the ElasticSearch index.
- **Flexible Document Retrieval:** Retrieve documents based on relevancy to the query, ensuring users get the most pertinent information.

## Getting Started

These instructions will guide you through the setup and deployment of Squirro Chatbot on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Elasticsearch: The user needs to setup and start an Elastic Search Server. [Setup Link](https://www.elastic.co/downloads/elasticsearch) 
- OpenAI API Access: Open AI API Access is required only for generating an answer to queries. Search functionality works without the OpenAI API. [Link](https://openai.com/blog/openai-api)

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/tushar7797/squirro_chatbot.git
   cd squirro_chatbot
   ```

2. Install the repository in development mode

    ```sh
   pip install -e .
   ```
    We strongly suggest creating a conda environment and then installing the repository.

3. Setup Environment Variables

    You will need to set environment variables for ElasticSearch and OpenAI API credentials:
    ```sh
    export ELASTICSEARCH_USERNAME='your_username'
    export ELASTICSEARCH_PASSWORD='your_password'
    export OPENAI_API_KEY='your_openai_api_key'
    export OPENAI_ORGANIZATION='your_openai_organization'
    ```

4. Update App configurations accordingly in ```squirro_chatbot/app_config.py```

5. Running the Application

    ```sh
    python3 app.py
    ```

## Usage

The application provides the following endpoints:

### Creating Documents

**POST /documents/**: Submit a document for indexing.

  ```sh
  curl -X POST http://127.0.0.1:5000/documents/ -H "Content-Type: application/json" -d '{"text": "Water is H2O."}'
  ```

  ### Retrieving Documents

**GET /documents/<doc_id>/**: Retrieve a document by its ID.

  ```sh
curl http://127.0.0.1:5000/documents/c41cbbf2c21619e1d51dd729dbd9dd73693672ac0e358bfcda467827ba41bdf7
  ```

  ### Searching for Documents

**GET /search/**: Search documents based on a query.

  ```sh
curl "http://127.0.0.1:5000/search/?query=what%20is%20water?&top_k=2"
  ```

### Generating Answers

**GET /generate_answer/**: Generate an answer to a query using the most relevant documents. This endpoint uses the same query parameters as the search endpoint.

  ```sh
  curl "http://127.0.0.1:5000/generate_answer/?query=what%20is%20water?"
  ```

## Relevance Tuning with FIQA

We also provide some experiments that we conducted on Relevance Tuning with the [Financial Opinion Mining and Question Answering](https://sites.google.com/view/fiqa/home) dataset. These experiments can be found in fiqa_notebooks. We also explain all the experiments and the results in the documentation.

## Run the Tests

The tests for relevant functionalities are provided in the files "squirro_chatbot/app_test.py" and "squirro_chatbot/models/openai_chat_model_test.py". One can either run the tests by running each of these with python3 or install pytest and run 

  ```python
   python3 -m pytest . 
  ```
