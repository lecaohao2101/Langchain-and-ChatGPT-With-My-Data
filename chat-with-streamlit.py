import os
import dotenv
import streamlit as st
from pymongo import MongoClient
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

dotenv.load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.transcripts
transcript_collection = db.transcripts


def main():
    st.set_page_config(page_title="Ask your Documents")
    st.header("Ask your Documents 💬")

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
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # show user input
    user_question = st.text_input("Ask a question about the transcribed documents:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)

        st.write(response)


if __name__ == "__main__":
    main()
