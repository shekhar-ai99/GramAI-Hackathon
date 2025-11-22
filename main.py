# main.py — GramAI FINAL with Dynamic CSS (Local / Gradio Live Auto Switch)

import gradio as gr
import torch
from torchvision import transforms, models
import json
import os
from gtts import gTTS

# -----------------------------------------
# Load config JSON
# -----------------------------------------
with open("config/diseases.json", "r", encoding="utf-8") as f:
    config = json.load(f)


# -----------------------------------------
# Auto model loader
# -----------------------------------------
def load_model(cfg):
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(cfg["classes"])
    model.classifier[3] = torch.nn.Linear(1024, num_classes)

    try:
        state = torch.hub.load_state_dict_from_url(cfg["model_url"], map_location="cpu")
        model.load_state_dict(state)
        print("Loaded:", cfg["model_url"])
    except:
        print("⚠ Warning: model weights not loaded")

    model.eval()
    return model


paddy_model = load_model(config["paddy"])
skin_model = load_model(config["skin"])


# -----------------------------------------
# Image transform
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# -----------------------------------------
# Odia Audio TTS
# -----------------------------------------
def speak_odia(text):
    path = "odia_voice.mp3"
    try:
        gTTS(text, lang="or").save(path)
    except:
        gTTS(text, lang="hi").save(path)
    return path


# -----------------------------------------
# Dynamic CSS Loader
# -----------------------------------------
is_gradio_live = "GRADIO_SERVER_PORT" in os.environ
css_file = "css/gradio_bg.css" if is_gradio_live else "css/local_bg.css"

with open(css_file, "r", encoding="utf-8") as f:
    css = f"<style>{f.read()}</style>"

print("Using CSS:", css_file)


# -----------------------------------------
# Prediction Function
# -----------------------------------------
def predict(img, mode):
    if img is None:
        return None, "ଫଟୋ ଦିଅନ୍ତୁ", "", None

    tensor = transform(img).unsqueeze(0)

    if "ଧାନ" in mode:
        cfg = config["paddy"]
        model = paddy_model
    else:
        cfg = config["skin"]
        model = skin_model

    classes = cfg["classes"]

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0]
        idx = prob.argmax().item()
        conf = prob[idx].item() * 100

    disease = classes[idx]
    details = cfg["details"][disease]

    color = "#198754" if "Healthy" in disease else "#dc3545"
    conf_bar_html = f"<div class='conf-bar' style='background:{color}; width:{conf}%'>Confidence: {conf:.1f}%</div>"

    odia_text = (
        f"ରୋଗ: {disease}\n"
        f"ବିବରଣୀ: {details['odia']['description']}\n"
        f"ଉପଚାର: {details['odia']['remedy']}\n"
        f"ମାତ୍ରା: {details['odia']['dosage']}\n"
        f"ସାବଧାନୀ: {details['odia']['warnings']}"
    )

    eng_text = (
        f"Disease: {disease} ({conf:.1f}%)\n"
        f"Description: {details['en']['description']}\n"
        f"Treatment: {details['en']['remedy']}\n"
        f"Dosage: {details['en']['dosage']}\n"
        f"Warnings: {details['en']['warnings']}"
    )

    audio_path = speak_odia(details["odia"]["remedy"])

    return conf_bar_html, odia_text, eng_text, audio_path


# -----------------------------------------
# UI Layout
# -----------------------------------------
with gr.Blocks(css=css, theme=gr.themes.Soft(), title="GramAI") as demo:

    if is_gradio_live:
        gr.HTML("<div class='bg-wrapper'>")
    else:
        gr.HTML("<div>")

    gr.HTML("""
        <div class='card'>
            <h1>ଗ୍ରାମଏଆଇ – GramAI</h1>
            <p style='text-align:center; font-size:20px; color:#0f5132'>
                Paddy & Skin Disease Detection • ଓଡ଼ିଆରେ ଉପଚାର
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='card'>")
            mode = gr.Radio(["Paddy / ଧାନ", "Skin / ଚର୍ମ"], label="Select")
            img_input = gr.Image(type="pil", label="Upload Image", height=350)
            gr.HTML("</div>")

        with gr.Column(scale=1):
            gr.HTML("<div class='card'>")
            btn = gr.Button("Analyze / ଚିହ୍ନଟ କରନ୍ତୁ")
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ ଫଳାଫଳ", lines=7)
            eng_out = gr.Textbox(label="English Result", lines=7)
            audio = gr.Audio(label="Odia Audio")
            gr.HTML("</div>")

    btn.click(predict, [img_input, mode], [conf_bar, odia_out, eng_out, audio])

    gr.HTML("</div>")


demo.launch(share=True)
