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

css = """
<style>
    /* FULL Odisha paddy background — visible everywhere */
    .gradio-container, html, body {
        background: url('https://i.postimg.cc/B6Xj8tk7/odisha-bg.jpg') no-repeat center center fixed !important;
        background-size: cover !important;
        min-height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    :root { --bg: transparent !important; --bg-dark: transparent !important; }

    /* Semi-transparent dark cards — paddy field shows through */
    .container {
        background: rgba(20, 20, 40, 0.88) !important;   /* Dark blue-purple, 88% opacity */
        backdrop-filter: blur(8px);                     /* Beautiful glass effect */
        border: 1px solid rgba(0, 255, 100, 0.3);
        border-radius: 24px;
        padding: 30px;
        margin: 20px auto;
        max-width: 1000px;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.6);
    }
    h1 {
        color: #00ff62 !important;
        text-shadow: 0 0 20px #00ff62;
        font-size: clamp(42px, 10vw, 72px);
    }
    .gr-button {
        background: linear-gradient(45deg, #ff6200, #ff8c00) !important;
        border: none !important;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(255, 98, 0, 0.5);
    }
    .confidence {
        background: linear-gradient(90deg, #ff6200, #ff8c00);
        height: 55px;
        border-radius: 30px;
        font-size: 26px;
        box-shadow: 0 5px 20px rgba(255, 98, 0, 0.6);
    }
    .gr-textbox, .gr-radio {
        background: rgba(10, 10, 30, 0.9) !important;
        color: #fff !important;
        border: 1px solid #00ff62;
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