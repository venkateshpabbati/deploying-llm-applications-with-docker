# Deploying LLM Applications with Docker

There are two main approaches to developing and deploying AI applications:
- **Fully Open Source**: This approach emphasizes privacy and data protection.
- **Fully Closed Source**: This method involves integrating multiple APIs and cloud services.
  
Both approaches have their advantages and disadvantages. In our case, we have chosen the second approach, where we will integrate multiple AI services. This allows us to build an AI application that is fast and takes only a few seconds to build and deploy. Our main focus is to reduce the Docker image size, which can be effectively achieved by integrating multiple AI services.
We will be building an all-purpose document Q&A chatbot that allows users to upload any documents and chat with it at real-time speed. It is quite similar to the Google’s NotebookLM. 

![Deploying LLM Applications with Docker](https://github.com/user-attachments/assets/1d82de55-3db6-4f70-8ff3-f2c46a5ed6ee)

Here are tools that we will be using in this project:
1. **Gradio Blocks**: For creating a user interface that allows users to upload any text document and chat with the document easily.
2. **LlamaCloud**: For parsing files and extracting text data into markdown style.
3. **MixedBread AI**: For converting loaded documents into embeddings and also converting chat messages into embeddings for context retrieval.
4. **Groq Cloud**: For accessing fast LLM responses. In this project, we will be using the llama-3.1-70b model.
5. **LlamaIndex**: For creating the RAG (Retrieval Augmented Generation) pipeline that orchestrates all of the AI services. The pipeline will use the uploaded file and user messages to generate context-aware answers.
6. **Docker**: For encapsulating the app, dependencies, environment, and configurations.
7. **Hugging Face Cloud**: We will push all the files to the Spaces repository, and Hugging Face will automatically build the image using the Dockerfile and deploy it to the server.
