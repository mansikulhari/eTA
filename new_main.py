import io
from collections import defaultdict
import pdfplumber
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from custom_embedding import JinaEmbeddings
from vector_db import VectorDB



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def extract_text_and_images_from_pdf(pdf_docs, text_image_mapping):
    raw_text = ""
    all_chunks = []

    for doc in pdf_docs:
        with pdfplumber.open(doc) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if not text:
                    continue

                text_chunks = get_text_chunks(text)
                all_chunks.extend(text_chunks)
                raw_text += text
                if not text_chunks:
                    continue
                images_meta = page.images

                for i, text_chunk in enumerate(text_chunks):
                    nearby_images = []

                    for img_meta in images_meta:
                        x0, y0, x1, y1 = img_meta['x0'], img_meta['y0'], img_meta['x1'], img_meta['y1']
                        image_stream = page.crop((x0, y0, x1, y1)).to_image()
                        img_byte_arr = io.BytesIO()
                        image_stream.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        nearby_images.append(img_byte_arr)

                    text_image_mapping[text_chunk].extend(nearby_images)
    print('num images:', len(text_image_mapping.values()))
    return all_chunks


def get_text_chunks(text: str) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = JinaEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    pipe = pipeline(
        "text-generation", model=llm, tokenizer=tokenizer, max_new_tokens=50,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=hf,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )

    return conversation_chain



def handle_userinput(user_question, text_image_mapping):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

    # Get the retrieved text chunk
    distances, indices = st.session_state.vector_store.search(user_question, k=3)
    context_chunks = [st.session_state.vector_store.get_text_chunk(idx) for idx in indices[0]]

    for context_chunk in context_chunks:
        print('context text chunk:', context_chunk, '\n\n')
        print('number of images:', len(text_image_mapping[context_chunk]), '\n\n')
        # Check if the retrieved text chunk has associated images
        if context_chunk in text_image_mapping:
            # Display context and images
            # st.write(f"**Context**: {context_chunk}")
            associated_images = text_image_mapping[context_chunk]
            for img_data in associated_images:
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption="Associated Image")


def main():
    # i think i'll just have to use 2 vector DBs so that I can retrieve relevant chunks for images
    # if i want to improve the model so that I can embed images in between text rather than at the end, I think
    # then I won't need two; either way, I think the result will end up being better by using langchain because of
    # the built-in prompt engineering and memory management; not the worst thing in the world that I built the other
    # method since it might be useful for other things down the road for other features

    text_image_map = defaultdict(list)
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorDB(JinaEmbeddings())



    st.header("Chat with PDFs :robot_face:")

    if "session_ended" not in st.session_state:
        st.session_state.session_ended = False

    # Check if session has ended
    if st.session_state.session_ended:
        st.warning("Session ended. Please refresh the page if you wish to start again.")
        return

    user_question = st.text_input("Ask questions about your documents:")

    if user_question:
        if user_question.strip().lower() == "exit":
            st.session_state.session_ended = True
            st.warning("Session ended. Please refresh the page if you wish to start again.")
            return
        handle_userinput(user_question, text_image_map)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                text_chunks = extract_text_and_images_from_pdf(pdf_docs, text_image_map)

                # text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store.update_vectorstore(text_chunks)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )


if __name__ == '__main__':
    main()




