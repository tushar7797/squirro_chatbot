## Command to Create a document

curl -X POST http://127.0.0.1:5000/documents/ -H "Content-Type: application/json" -d '{"text": "This is a test document"}' 

## Command to Retrieve a document

curl http://127.0.0.1:5000/documents/c41cbbf2c21619e1d51dd729dbd9dd73693672ac0e358bfcda467827ba41bdf7

## Command to retrieve relevant documents

curl "http://127.0.0.1:5000/search/?query=test%20document&top_k=2"