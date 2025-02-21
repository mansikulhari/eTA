import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import time


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# text = "A german Shepherd dog standing in a field of grass"
text = "what is a convolutional neural network?"

# image_url = "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg"
# image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png"
image_url = "https://www.varsitytutors.com/assets/vt-hotmath-legacy/hotmath_help/topics/graphing-cosine-function/cos-graph.gif"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Preprocess the text and the image
start = time.time()
text_inputs = processor(text, return_tensors="pt", padding=True, truncation=True)
image_inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)

# Get the embeddings
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    image_features = model.get_image_features(**image_inputs)


text_features = text_features.squeeze(0)  # Remove the batch dimension
image_features = image_features.squeeze(0)  # Remove the batch dimension
end = time.time()
similarity = F.cosine_similarity(text_features, image_features, dim=0).item()


print(f"Cosine similarity: {similarity:.4f}")
print('Duration:', end - start)


