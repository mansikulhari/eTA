from flask import Flask, request, jsonify
from utils import get_conversation_chain, get_vectorstore, parse_notes, parse_subtitles, generate_yt_url
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from dotenv import load_dotenv
from collections import defaultdict


load_dotenv()
app = Flask(__name__)

# Parse documents and initialize the conversation chain and other resources
texts, txt_to_img = parse_notes()
subtitles, sub_to_timestamp = parse_subtitles()
texts.extend(subtitles)

print('Instantiating vectorstore')
vectorstore = get_vectorstore(texts)

print('Instantiating LLM')
llm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

print('Instantiating tokenizer')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

print('Instantiating conversation chain')
conversation = get_conversation_chain(vectorstore, llm, tokenizer)

# Initialize mappings for all requests
associated_images = defaultdict(list)
associated_timestamps = defaultdict(list)


def format_chat_history(chat_history, associated_images, associated_timestamps):
    formatted_chat_history = []
    for i in range(len(chat_history)):
        if i % 2 == 0:
            # User messages
            formatted_chat_history.append({
                "type": "user",
                "content": chat_history[i].content
            })
        else:
            # Bot messages
            timestamps = associated_timestamps.get(chat_history[i].content, [])
            urls = [generate_yt_url(timestamp[0], timestamp[1]) for timestamp in timestamps]
            formatted_chat_history.append({
                "type": "bot",
                "content": chat_history[i].content,
                "images": associated_images.get(chat_history[i].content, []),
                "timestamps": timestamps,
                "urls": urls
            })

    return formatted_chat_history



@app.route('/query', methods=['POST'])
def query():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Perform the search
    docs = vectorstore.similarity_search(user_question, k=3)
    relevant_context = [doc.page_content for doc in docs]

    # Get the conversation response
    response = conversation({'question': user_question})

    bot_response = response['chat_history'][-1].content

    for context_text_chunk in relevant_context:
        if context_text_chunk in txt_to_img:
            # Map images to the context they are associated with
            associated_images[bot_response].extend(txt_to_img[context_text_chunk])
        if context_text_chunk in sub_to_timestamp:
            # Map timestamps to the context they are associated with
            associated_timestamps[bot_response].append(sub_to_timestamp[context_text_chunk])


    # Return the relevant data
    return jsonify({
        "conversation": format_chat_history(response['chat_history'], associated_images, associated_timestamps)
    })



if __name__ == '__main__':
    app.run(debug=True)  # You can remove debug=True in production
