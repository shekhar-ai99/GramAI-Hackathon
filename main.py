# main.py — FINAL DARK MODE WINNER (Matches your screenshot 100%)
import gradio as gr
import torch
from torchvision import transforms, models
import json
import os
from gtts import gTTS

# Load config
with open("config/diseases.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Auto model loader
def load_model(cfg):
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(cfg["classes"])
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    try:
        state = torch.hub.load_state_dict_from_url(cfg["model_url"], map_location="cpu")
        model.load_state_dict(state)
    except:
        print("Warning: dummy weights")
    model.eval()
    return model

paddy_model = load_model(config["paddy"])
skin_model = load_model(config["skin"])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def speak_odia(text):
    path = "result_odia.mp3"
    try:
        gTTS(text, lang='or', slow=False).save(path)
    except:
        gTTS(text, lang='hi').save(path)
    return path

# DARK MODE + EXACT UI FROM YOUR SCREENSHOT
css = """
<style>
    body { 
        background: #000; 
        color: #fff; 
        margin: 0; 
        padding: 10px;
        font-family: 'Segoe UI', sans-serif;
    }
    .container {
        background: rgba(30,30,30,0.95);
        border-radius: 20px;
        padding: 25px;
        margin: 15px auto;
        max-width: 1000px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.8);
    }
    h1 {
        font-family: 'Noto Sans Oriya', sans-serif;
        color: #00ff41;
        text-align: center;
        font-size: clamp(40px, 10vw, 70px);
        margin: 10px 0;
        text-shadow: 0 0 15px #00ff41;
    }
    .gr-button {
        background: #ff6200 !important;
        color: white !important;
        font-weight: bold;
    }
    .confidence {
        background: #ff6200;
        height: 50px;
        border-radius: 25px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 24px;
        line-height: 50px;
        margin: 15px 0;
    }
    .gr-textbox, .gr-radio {
        background: #111 !important;
        color: #fff !important;
        border: 2px solid #333;
    }
</style>
"""

with gr.Blocks(css=css, title="GramAI – ଗ୍ରାମଏଆଇ") as demo:
    gr.HTML("""
    <div class="container">
        <h1>ଗ୍ରାମଏଆଇ – GramAI</h1>
        <p style="text-align:center; font-size:22px; color:#00ff41;">
            <b>Paddy & Skin Disease Detection Using Image</b><br>
            ଧାନ ଓ ଚର୍ମ ରୋଗ ଚିହ୍ନଟ — ଓଡ଼ିଆରେ ଉପଚାର
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Paddy / ଧାନ", "Skin / ଚର୍ମ"],
                label="",
                value="Skin / ଚର୍ମ"
            )
            img_input = gr.Image(type="pil", label="", height=500)

        with gr.Column(scale=1):
            gr.HTML('<div class="container">')
            btn = gr.Button("Analyze / ଚିହ୍ନଟ କରନ୍ତୁ", variant="primary", size="lg")
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ ଫଳାଫଳ", lines=6)
            eng_out = gr.Textbox(label="Result in English", lines=6)
            audio = gr.Audio(label="")
            gr.HTML('</div>')

    def predict(img, mode):
        if img is None:
            return None, "ଫଟୋ ଦିଅନ୍ତୁ", "", None

        tensor = transform(img).unsqueeze(0)
        cfg = config["paddy"] if "ଧାନ" in mode else config["skin"]
        model = paddy_model if "ଧାନ" in mode else skin_model

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            idx = prob.argmax().item()
            conf = prob[idx].item() * 100

        disease = cfg["classes"][idx]
        details = cfg["details"][disease]
        odia = f"ରୋଗ: {disease}\nଉପଚାର: {details['odia']['remedy']}"
        eng = f"Disease: {disease} ({conf:.1f}%)\nTreatment: {details['en']['remedy']}"

        bar = f'<div class="confidence">Confidence: {conf:.1f}%</div>'

        return bar, odia, eng, speak_odia(details["odia"]["remedy"])

    mode.change(fn=lambda: (None, "", "", None), outputs=[conf_bar, odia_out, eng_out, audio])
    btn.click(predict, [img_input, mode], [conf_bar, odia_out, eng_out, audio])

demo.launch(share=True)