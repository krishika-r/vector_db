# vector_db

Steps to install Milvus in Ubuntu Windows

1. install docker - https://docs.docker.com/engine/install/ubuntu/

2. install Milvus - https://milvus.io/docs/install_standalone-docker.md 

 - wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml
 - sudo docker-compose up -d = rename the downloaded file
 - sudo docker compose ps
 - docker port milvus-standalone 19530/tcp(connect to milvus)

## Document Processing Workflow
Step 1: PDF Extraction using Langchain
Utilize Langchain for extracting text from PDF documents.

Step 2: Chunking
Implement a chunking process to break down the extracted text into manageable pieces.

Step 3: Embedding
Apply embedding techniques to convert chunks of text into numerical representations.

Step 4: Storing in Milvus
Utilize Milvus to store the embedded vectors efficiently.

Step 5: Retrieving Data using Milvus Metrics
Retrieve and analyze data from Milvus using metrics for performance evaluation.

By following this comprehensive workflow, one can efficiently process, embed, store, and retrieve data using Milvus, creating a robust and scalable solution for our vector database needs.


