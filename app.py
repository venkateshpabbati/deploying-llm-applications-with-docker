import os

import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
#from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse

# API keys
llama_cloud_key = os.environ.get("LLAMAINDEX_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
#mxbai_key = os.environ.get("MIXEDBREAD_API_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")
if not (llama_cloud_key and groq_key and cohere_key): #mxbai_key):
    raise ValueError(
        "API Keys not found! Ensure they are passed to the Docker container."
    )

# models name
llm_model_name = "llama-3.3-70b-versatile"
#embed_model_name = "mixedbread-ai/mxbai-embed-large-v1"
embed_model_name = "embed-english-v3.0"

# Initialize the parser
parser = LlamaParse(api_key=llama_cloud_key, result_type="markdown")

# Define file extractor with various common extensions
file_extractor = {
    ".pdf": parser, ".docx": parser, ".doc": parser,
    ".csv": parser, ".xlsx": parser, ".pptx": parser,
    ".html": parser, ".jpg": parser, ".jpeg": parser,
    ".png": parser, ".webp": parser, ".svg": parser,
    ".txt": parser, ".json": parser, ".xml": parser,
    ".md": parser, ".yml": parser, ".yaml": parser, ".log": parser,
    ".ppt": parser, ".epub": parser, ".odt": parser, ".ods": parser,
    ".odp": parser, ".rtf": parser, ".tex": parser, ".ps": parser,
    ".docm": parser, ".dotx": parser, ".dotm": parser, ".wps": parser,
    ".wpd": parser, ".wp": parser, ".pages": parser, ".numbers": parser,
    ".key": parser, ".indd": parser, ".ai": parser, ".sketch": parser,
    ".fig": parser, ".svgz": parser, ".tiff": parser, ".tif": parser,
    ".bmp": parser, ".raw": parser, ".heic": parser, ".webm": parser,
    }

# Initialize the embedding model
#embed_model = MixedbreadAIEmbedding(api_key=mxbai_key, model_name=embed_model_name)
embed_model = CohereEmbedding(
    api_key=cohere_key, model_name=embed_model_name) #, temperature=0.2, max_retries=3)

# Initialize the LLM
llm = Groq(model=llm_model_name, api_key=groq_key)


# File processing function
def load_files(file_path: str):
    global vector_index
    if not file_path:
        return "No file path provided. Please upload a file."
    
    valid_extensions = ', '.join(file_extractor.keys())
    if not any(file_path.endswith(ext) for ext in file_extractor):
        return f"The parser supports only: {valid_extensions}"

    document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    vector_index = VectorStoreIndex.from_documents(document, embed_model=embed_model)
    print(f"Parsing completed for: {file_path}")
    filename = os.path.basename(file_path)
    return f"Ready to provide responses based on: {filename}"

# Respond function
def respond(message, history):
    try:
        # Use the preloaded LLM
        query_engine = vector_index.as_query_engine(streaming=True, llm=llm)
        streaming_response = query_engine.query(message)

        partial_text = ""
        for new_text in streaming_response.response_gen:
            partial_text += new_text
            # Build history for each response step
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial_text},
            ]
            # Yield an empty string to cleanup the message textbox and the updated conversation history
            #yield partial_text
    except (AttributeError, NameError):
        print("Error: No vector index loaded.")
        #yield "Please upload the file to begin chat."
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload the file to begin chat."},
        ]

# Clear function
def clear_state():
    global vector_index
    vector_index = None
    return [None, None, None]

# UI Setup
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="green",
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Poppins")],
    ),
    css="footer {visibility: hidden}",
) as demo:
    gr.Markdown("#Doc Q&A 🤖📃")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                file_count="single", type="filepath", label="Upload Document"
            )
            with gr.Row():
                btn = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear")
            output = gr.Textbox(label="Status")

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                type="messages",
                chatbot=gr.Chatbot(height=300,type="messages"),
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(
                    placeholder="Ask questions about the uploaded document!",
                    container=False,
                ),
            )

    # Set up Gradio interactions
    btn.click(fn=load_files, inputs=file_input, outputs=output)
    clear.click(
        fn=clear_state,  # Use the clear_state function
        outputs=[file_input, output],
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()