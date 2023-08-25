import streamlit as st
import replicate
import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from PIL import Image

pinecone.init(api_key = st.secrets["api_key"], environment = st.secrets["environment"])

# App title
st.set_page_config(page_title="Sathya Sai Echos")

# urllib.request.urlretrieve(
#   'https://drive.google.com/file/d/1-K77NZZ2vwUzB1LWZshz53uQFx1ljDE-/view?usp=sharing',
#    "sse.png")
# original_image = Image.open("sse.png")

# # Resize the image to a smaller dimension
# width, height = 370, 300  

# # Adjust the dimensions as needed
# resized_image = original_image.resize((width, height))

# # Save the resized image
# resized_image.save("resized_image.png")

# Replicate Credentials
with st.sidebar:
    st.title('Sathya Sai Echos')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello Sai Bangaru!! What brings you here today ? 😊"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

chat_history = []

def clear_chat_history():
    chat_history = []
    st.session_state.messages = [{"role": "assistant", "content": "Are there any other things that you want me to help you with? 😊"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response

def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = swamioutput(string_dialogue, prompt_input)
    return output

def swamioutput(string_dialogue, prompt_input):
    # Using HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    
    # Setting up Pinecone vector database
    index_name = "saidb"
    index = pinecone.Index(index_name)
    vectordb = Pinecone.from_existing_index(index_name, embeddings)

    # Initialize Replicate Llama2 Model
    llm = Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
                        input={"temperature": 0.75, "max_length": 800})

    # Set up the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectordb.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True
    )

    # Start chatting with the chatbot
    result = qa_chain({'question': f"{prompt_input}", 'chat_history': chat_history})
    chat_history.append((prompt_input, result['answer']))

    return result['answer']

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
