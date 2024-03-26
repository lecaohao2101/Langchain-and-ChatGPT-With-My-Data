import os
import dotenv
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from pymongo import MongoClient

# import API key from .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.transcripts
transcript_collection = db.transcripts

# speech to text
def transcribe(input_file):
    transcript = openai.Audio.transcribe("whisper-1", input_file)
    return transcript

# save file upload
def save_file(audio_bytes, file_name):
    with open(file_name, "wb") as f:
        f.write(audio_bytes)

# read file upload and transcribe
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = transcribe(audio_file)

    return transcript["text"]

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Documents")
    st.header("Ask your Documents 💬")

    # upload audio files (optional)
    uploaded_files = st.file_uploader("Upload your audio files (optional)", type=["mp4"], accept_multiple_files=True)

    # transcribe and save transcript files if uploaded
    transcript_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # get name_file_input and remove extension
            name_file_input = os.path.splitext(uploaded_file.name)[0]

            # render type input_file /mp4
            audio_file_name = f"{name_file_input}.{uploaded_file.type.split('/')[1]}"

            # create .txt
            transcript_file_name = f"{name_file_input}_transcript.txt"

            # save audio file
            save_file(uploaded_file.read(), audio_file_name)

            # contains text of audio_file_name
            transcript_text = transcribe_audio(audio_file_name)

            # Save transcript text in MongoDB
            transcript_collection.insert_one({"name": transcript_file_name, "text": transcript_text})

            transcript_files.append({"name": transcript_file_name, "text": transcript_text})

    # Get all transcript files from MongoDB
    mongo_transcripts = list(transcript_collection.find({}, {"_id": 0, "text": 1}))
    all_text = "\n".join([transcript["text"] for transcript in mongo_transcripts])

    # Process transcript files
    process_transcript_files(all_text)

def process_transcript_files(all_text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Update the parameter names

    # show user input
    user_question = st.text_input("Ask a question about the transcribed documents:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(response)

        st.write(response)

if __name__ == "__main__":
    main()