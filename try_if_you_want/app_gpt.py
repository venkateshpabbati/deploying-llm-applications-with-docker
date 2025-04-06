import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse

# API keys
llama_cloud_key = os.environ.get("LLAMAINDEX_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")
if not (llama_cloud_key and groq_key and cohere_key):
    raise ValueError("API Keys not found! Ensure they are passed to the Docker container.")

# Model names
llm_model_name = "llama-3.3-70b-versatile"
embed_model_name = "embed-english-v3.0"

# Initialize the parser
parser = LlamaParse(api_key=llama_cloud_key, result_type="markdown")

# Define file extractor
file_extractor = {
    ext: parser for ext in [
        ".pdf", ".docx", ".doc", ".csv", ".xlsx", ".pptx", ".html", ".jpg",
        ".jpeg", ".png", ".webp", ".svg", ".txt", ".json", ".xml", ".md", ".yml",
        ".yaml", ".log", ".ppt", ".epub", ".odt", ".ods", ".odp", ".rtf", ".tex",
        ".ps", ".docm", ".dotx", ".dotm", ".wps", ".wpd", ".wp", ".pages",
        ".numbers", ".key", ".indd", ".ai", ".sketch", ".fig", ".svgz", ".tiff",
        ".tif", ".bmp", ".raw", ".heic", ".webm"
    ]
}

# Initialize the embedding model
embed_model = CohereEmbedding(api_key=cohere_key, model_name=embed_model_name)

# Initialize the LLM
llm = Groq(model=llm_model_name, api_key=groq_key)

# File processing function
def load_files(file_paths, state):
    if not file_paths:
        return "No files uploaded.", state

    documents = []
    for path in file_paths:
        if not any(path.endswith(ext) for ext in file_extractor):
            continue
        doc = SimpleDirectoryReader(input_files=[path], file_extractor=file_extractor).load_data()
        documents.extend(doc)

    state = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    filenames = ", ".join(os.path.basename(path) for path in file_paths)
    return f"Ready to provide responses based on: {filenames}", state

# Respond function
def respond(message, history, state):
    try:
        if state is None:
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please upload files first."}], state

        query_engine = state.as_query_engine(streaming=True, llm=llm)
        streaming_response = query_engine.query(message)

        partial_text = ""
        for new_text in streaming_response.response_gen:
            partial_text += new_text
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial_text},
            ], state

    except Exception as e:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"An error occurred: {e}"},
        ], state

# Clear function
def clear_state():
    return [None, None, None]

# UI Setup
with gr.Blocks(theme=gr.themes.Default(primary_hue="green", secondary_hue="blue", font=[gr.themes.GoogleFont("Poppins")]), css="footer {visibility: hidden}") as demo:
    state = gr.State()
    gr.Markdown("#Doc Q&A 🤖📃")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", type="filepath", label="Upload Documents")
            with gr.Row():
                btn = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear")
            output = gr.Textbox(label="Status")

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                type="messages",
                chatbot=gr.Chatbot(height=300, type="messages"),
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(placeholder="Ask questions about the uploaded documents!", container=False),
                additional_inputs=[state],
                additional_outputs=[state]
            )

    # Gradio interactions
    btn.click(fn=load_files, inputs=[file_input, state], outputs=[output, state])
    clear.click(fn=clear_state, outputs=[file_input, output, state])

if __name__ == "__main__":
    demo.launch()
