# GramAI Swasthya O Krushi Sahayak – Northern Odisha AI Hackathon 2025
# One app → Paddy Disease + Skin Disease + Odia Voice

import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from gtts import gTTS
import os

print("Loading models... (first run takes 2–3 mins)")

# === Paddy Model ===
paddy_model = models.mobilenet_v3_small(pretrained=False)
paddy_model.classifier[3] = torch.nn.Linear(1024, 5)
paddy_model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/spaces/fffiloni/paddy-disease-classification/resolve/main/paddy_model.pth",
    map_location="cpu"
))
paddy_model.eval()
paddy_classes = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Blast", "Healthy", "Tungro"]

# === Skin Model ===
skin_model = models.mobilenet_v3_small(pretrained=False)
skin_model.classifier[3] = torch.nn.Linear(1024, 7)
skin_model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/spaces/ahmedshahriar/Skin_Disease/resolve/main/skin_model.pth",
    map_location="cpu"
))
skin_model.eval()
skin_classes = ["Acne", "Eczema", "Psoriasis", "Ringworm (Dadru)", "Scabies", "Fungal Infection", "Healthy Skin"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Remedies in Odia ===
paddy_remedies = { ... }   # ← copy from previous message (same dictionary)
skin_remedies = { ... }     # ← copy from previous message

def predict_image(img, mode):
    # ← copy the exact same function from previous message
    # (the one with gTTS saving "result_odia.mp3")
    pass

# === Gradio Interface ===
with gr.Blocks(title="GramAI") as demo:
    gr.Markdown("# 🌾🩺 GramAI – ଗ୍ରାମଏଆଇ")
    gr.Markdown("### ଓଡ଼ିଆରେ ଧାନ + ଚର୍ମ ରୋଗ ଚିହ୍ନଟ | Northern Odisha Hackathon 2025")
    mode = gr.Radio(["🌾 Paddy / ଧାନ", "🩺 Skin / ଚର୍ମ"], label="ବାଛନ୍ତୁ | Choose:")
    img = gr.Image(type="pil")
    btn = gr.Button("🔍 ଦେଖନ୍ତୁ | Analyze")
    out_text = gr.Textbox(label="ଉତ୍ତର | Result")
    out_audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି | Listen")
    btn.click(predict_image, [img, mode], [out_text, out_audio])

demo.launch(share=True)