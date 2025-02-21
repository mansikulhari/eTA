css = '''
<style>
.element-container,.stMarkdown {
    width: 750px;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    width: 100%;
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
    justify-content: space-around;
}
.chat-message .avatar img {
  max-width: 50px;
  max-height: 50px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  padding: 0 .5rem;
  color: #fff;
}
.context-image {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Chatbot_img.png" style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{message}</div>
    <div class="context-image">{img_html}</div>
    <div class="context">{context_html}</div>
</div>
'''


user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png" style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;>
    </div>    
    <div class="message" >{{MSG}}</div>
</div>
'''