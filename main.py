import streamlit as st
from htmlTemplates import css, bot_template, user_template
import requests
from utils import timestamp_to_readable, get_image_as_base64



def init_streamlit():
    st.set_page_config(page_title="Chat with Online Courses", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with Online Courses :robot_face:")

    if "session_ended" not in st.session_state:
        st.session_state.session_ended = False

    # Check if session has ended
    if st.session_state.session_ended:
        st.warning("Session ended. Please refresh the page if you wish to start again.")
        return


API_URL = "http://localhost:5000/query"


def main():
    init_streamlit()

    user_question = st.text_input("Ask questions about your online course:")
    if user_question:
        # Send the user question to the Flask API and get the response
        response = requests.post(API_URL, json={"question": user_question})
        if response.status_code == 200:
            # Parse the response
            data = response.json()
            conversation = data.get('conversation', [])

            # Update the session state for chat history
            st.session_state.chat_history = conversation

            # Display messages
            for message in conversation:
                if message['type'] == 'user':
                    st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                elif message['type'] == 'bot':
                    img_html = ""
                    context_html = ""

                    # Check if there are associated images for this message
                    seen_images = set()
                    for img_path in message.get('images', []):

                        if img_path not in seen_images:
                            img_base64 = get_image_as_base64(img_path)
                            img_html += f'<img src="data:image/png;base64,{img_base64}" style="height: 150px; width:auto; margin:0 10px; border-radius: 10px;">'
                            seen_images.add(img_path)

                    # Check if there are associated timestamps for this message
                    seen_timestamps = set()
                    lecture_urls = message.get('urls', [])
                    for i, (lecture_name, start, end) in enumerate(message.get('timestamps', [])):
                        if lecture_name not in seen_timestamps:
                            start = timestamp_to_readable(start)
                            end = timestamp_to_readable(end)
                            context_html += f'<div class="context"><a href="{lecture_urls[i]}"><strong>{lecture_name}</strong></a></div>'
                            seen_timestamps.add(lecture_name)

                    # Format the bot message with associated images and context
                    bot_message_html = bot_template.format(message=message["content"], img_html=img_html,
                                                           context_html=context_html)

                    # Write the bot message, including images and timestamps, to the frontend
                    st.write(bot_message_html, unsafe_allow_html=True)
        else:
            st.error("Failed to get a response from the server")



if __name__ == "__main__":
    main()


