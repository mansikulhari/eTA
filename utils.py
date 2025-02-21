import base64
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores.faiss import FAISS
from transformers import pipeline
import io
from collections import defaultdict
from langchain.embeddings.openai import OpenAIEmbeddings



# define global variables
VIDEO_IDS = {
    "lecture_0": "WbzNRTTrX0g",
    "lecture_1": "HWQLez87vqM",
    "lecture_2": "D8RRq3TbtHU",
    "lecture_3": "qK46ET1xk2A",
    "lecture_4": "-g0iJjnO2_w",
    "lecture_5": "J1QD9hLDEDY",
    "lecture_6": "QAZc9xsQNjQ",
}

def get_text_chunks(text, chunk_size=200, chunk_overlap=40):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def custom_text_splitter(text, chunk_size=200, chunk_overlap=40):
    chunks = []
    index = 0

    while index < len(text):
        # If we are not at the beginning and we have enough text left,
        # then we can consider the overlap
        if index > 0 and len(text) - index > chunk_overlap:
            index -= chunk_overlap

        # Find the end of the chunk, considering word breaks
        end = min(index + chunk_size, len(text))
        if end < len(text):
            while end > index and text[end] not in " \n":
                end -= 1

        # If we didn't find a space or newline, just hard cut at the chunk_size
        if end == index:
            end = index + chunk_size

        chunks.append(text[index:end])
        index = end

    return chunks


def extract_text_and_images_from_pdf(pdf_docs, text_image_mapping):
    raw_text = ""
    for doc in pdf_docs:
        with pdfplumber.open(doc) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                text_chunks = get_text_chunks(text)
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

    return raw_text


def get_conversation_chain(vectorstore, llm, tokenizer):
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
            search_type="similarity", search_kwargs={"k": 3}),
        memory=memory,
    )

    return conversation_chain


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def parse_notes():
    print('Parsing Notes')
    chunks = []
    txt_to_img = defaultdict(list)
    for i in range(7):
        with open(f'new_scraped_data/notes/lecture_{i}/notes.txt', 'r') as f:
            for line_num in range(20):
                f.readline()
            chunk = ""
            for line in f:
                if "[Image:" in line:
                    cut_chunks = get_text_chunks(chunk, chunk_size=200, chunk_overlap=40)
                    chunks.extend(cut_chunks)
                    start = line.find("./")
                    img_path = line[start:]
                    for cut_chunk in cut_chunks:
                        txt_to_img[cut_chunk].append(img_path.rstrip('\n'))
                    chunk = ""
                else:
                    chunk += line

    return chunks, txt_to_img


def parse_subtitles():
    print('Parsing Subtitles')
    chunks = []
    sub_to_timestamp = defaultdict(tuple)
    for i in range(7):
        lecture_name = f"lecture_{i}"
        with open(f'new_scraped_data/subtitles/{lecture_name}/subtitles.txt', 'r') as f:
            data = f.read().split('\n\n')
            curr = "00:00:00,000"
            chunk = ""
            for subtitle in data:
                if subtitle == "":
                    continue
                num, times, *lines = subtitle.split('\n')
                try:
                    start, end = times.split(' --> ')
                except:
                    print('not enough values to unpack:', lecture_name, times)
                end = end
                chunk += " ".join(lines)
                if time_to_seconds(end) - time_to_seconds(curr) >= 30:
                    cut_chunks = get_text_chunks(chunk, chunk_size=200, chunk_overlap=40)
                    chunks.extend(cut_chunks)
                    for cut_chunk in cut_chunks:
                        sub_to_timestamp[cut_chunk] = (lecture_name, curr, end)
                    curr = end
                    chunk = ""

    return chunks, sub_to_timestamp


def time_to_seconds(time: str) -> float:
    hours, minutes, seconds = time.split(":")
    seconds, milliseconds = seconds.split(",")

    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def timestamp_to_readable(timestamp):
    # Split the timestamp into its parts
    time_parts = timestamp.split(',')
    hours, minutes, seconds = time_parts[0].split(':')

    # Convert each part into an integer
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    # Create a human-readable format
    readable = f"{hours}:{minutes}:{seconds}"

    return readable


def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def generate_yt_url(lecture_name, start):
    start = timestamp_to_readable(start)
    video_id = VIDEO_IDS[lecture_name]
    hours, minutes, seconds = start.split(':')

    return f"https://www.youtube.com/watch?v={video_id}&t={hours}h{minutes}m{seconds}s"



