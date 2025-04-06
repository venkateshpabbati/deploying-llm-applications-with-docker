import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API keys
llama_cloud_key = os.environ.get("LLAMAINDEX_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")

if not (llama_cloud_key and groq_key and cohere_key):
    raise EnvironmentError("One or more API keys are missing. Please check environment variables.")

# Model names
llm_model_name = "llama-3.3-70b-versatile"
embed_model_name = "embed-english-v3.0"

# Initialize the parser
parser = LlamaParse(api_key=llama_cloud_key, result_type="markdown")

# File extractor
file_extractor = {ext: parser for ext in [
    ".pdf", ".docx", ".doc", ".csv", ".xlsx", ".pptx", ".html", ".jpg", ".jpeg", ".png", ".webp", ".svg",
    ".txt", ".json", ".xml", ".md", ".yml", ".yaml", ".log", ".ppt", ".epub", ".odt", ".ods", ".odp", ".rtf",
    ".tex", ".ps", ".docm", ".dotx", ".dotm", ".wps", ".wpd", ".wp", ".pages", ".numbers", ".key", ".indd",
    ".ai", ".sketch", ".fig", ".svgz", ".tiff", ".tif", ".bmp", ".raw", ".heic", ".webm"
]}

# Embedding and LLM initialization
embed_model = CohereEmbedding(api_key=cohere_key, model_name=embed_model_name)
llm = Groq(model=llm_model_name, api_key=groq_key)

# Load file and build index
def load_files(file_path: str):
    if not file_path:
        return "No file path provided. Please upload a file.", None

    if not any(file_path.lower().endswith(ext) for ext in file_extractor):
        valid_extensions = ', '.join(file_extractor.keys())
        return f"Unsupported file type. Supported types: {valid_extensions}", None

    try:
        documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
        vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        filename = os.path.basename(file_path)
        logging.info(f"Parsing completed for: {file_path}")
        return f"Ready to provide responses based on: {filename}", vector_index
    
    except Exception as e:
        logging.error(f"Error loading/indexing file: {e}")
        return f"Error processing file: {e}", None
    

# Chat response function
def respond(message, history, vector_index):
    if vector_index is None:
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please upload a file to begin chat."}], None
        return

    try:
        query_engine = vector_index.as_query_engine(streaming=True, llm=llm)
        streaming_response = query_engine.query(message)

        partial_text = ""
        for new_text in streaming_response.response_gen:
            partial_text += new_text
            yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": partial_text}], vector_index
    except Exception as e:
        logging.error(f"Error during query: {e}")
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"An error occurred: {e}"}], vector_index
        

# Reset function
def clear_state():
    return None, "", [], None

# Gradio UI
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="green",
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Poppins")],
    ),
    css="footer {visibility: hidden}",
) as demo:
    gr.Markdown("# Doc Q&A 🤖📃")
    vector_index_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="single", type="filepath", label="Upload Document")
            with gr.Row():
                btn = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear")
            output = gr.Textbox(label="Status")

        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(height=300, type="messages")
            chat_interface = gr.ChatInterface(
                fn=respond,
                inputs=["message", "history", vector_index_state],
                outputs=[chatbot_ui, vector_index_state],
                chatbot=chatbot_ui,
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(
                    placeholder="Ask questions about the uploaded document!",
                    container=False,
                ),
            )

    # Hook up interactions
    btn.click(fn=load_files, inputs=file_input, outputs=[output, vector_index_state])
    clear.click(fn=clear_state, outputs=[file_input, output, chatbot_ui, vector_index_state])

# Launch app
if __name__ == "__main__":
    demo.launch()
