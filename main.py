# main.py — GramAI Enhanced UI (Dynamic CSS, polished layout)
import gradio as gr
import torch
from torchvision import transforms, models
import json
import os
from gtts import gTTS
import time

# -------------------------
# Config + Models
# -------------------------
with open("config/diseases.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def load_model(cfg):
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(cfg["classes"])
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    try:
        state = torch.hub.load_state_dict_from_url(cfg["model_url"], map_location="cpu")
        model.load_state_dict(state)
        print("Loaded:", cfg["model_url"])
    except Exception as e:
        print("⚠ Warning loading weights:", e)
    model.eval()
    return model

paddy_model = load_model(config["paddy"])
skin_model = load_model(config["skin"])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def speak_odia(text):
    path = "odia_voice.mp3"
    try:
        gTTS(text, lang="or").save(path)
    except:
        gTTS(text, lang="hi").save(path)
    return path

# -------------------------
# Dynamic CSS handling
# -------------------------
is_gradio_live = "GRADIO_SERVER_PORT" in os.environ
css_file = "css/enhanced_gradio.css" if is_gradio_live else "css/enhanced_local.css"

with open(css_file, "r", encoding="utf-8") as f:
    css = f"<style>{f.read()}</style>"

print("Using CSS:", css_file)

# -------------------------
# Prediction (with UI-friendly delays & spinner control)
# -------------------------
def predict(image, mode):
    if image is None:
        return gr.update(visible=True), "କୃପୟା ଏକ ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ", "", None

    # simulate small UX delay for spinner micro-interaction (optional)
    time.sleep(0.25)

    tensor = transform(image).unsqueeze(0)
    cfg = config["paddy"] if "ଧାନ" in mode else config["skin"]
    model = paddy_model if "ଧାନ" in mode else skin_model

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    conf = float(probs[idx]) * 100.0
    disease = cfg["classes"][idx]
    details = cfg["details"].get(disease, {"odia":{"remedy":"ନାହିଁ","description":"N/A","dosage":"N/A","warnings":"N/A"},
                                            "en":{"remedy":"N/A","description":"N/A","dosage":"N/A","warnings":"N/A"}})

    color = "#198754" if "Healthy" in disease or "Healthy" in details["en"].get("remedy","") else "#dc3545"
    conf_html = f"""<div class="gauge"><svg viewBox="0 0 36 36" class="circular-chart" width="110" height="110">
      <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
      <path class="circle" stroke="{color}" stroke-dasharray="{conf:.1f},100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
      <text x="18" y="20.35" class="percentage">{conf:.1f}%</text>
    </svg></div>"""

    odia_text = (
        f"ରୋଗ: {disease}\n\n"
        f"ବର୍ଣ୍ଣନା: {details['odia'].get('description','N/A')}\n\n"
        f"ଉପଚାର: {details['odia'].get('remedy','N/A')}\n"
        f"ମାତ୍ରା: {details['odia'].get('dosage','N/A')}\n"
        f"ସାବଧାନୀ: {details['odia'].get('warnings','N/A')}"
    )

    eng_text = (
        f"Disease: {disease} ({conf:.1f}%)\n\n"
        f"Description: {details['en'].get('description','N/A')}\n\n"
        f"Treatment: {details['en'].get('remedy','N/A')}\n"
        f"Dosage: {details['en'].get('dosage','N/A')}\n"
        f"Warnings: {details['en'].get('warnings','N/A')}"
    )

    audio_path = speak_odia(details["odia"].get("remedy",""))

    return conf_html, odia_text, eng_text, audio_path

# -------------------------
# Enhanced UI layout
# -------------------------
with gr.Blocks(css=css, theme=gr.themes.Soft(), title="GramAI — Enhanced UI") as demo:

    # wrapper for gradio.live (uses .bg-wrapper in CSS) or plain div for local
    if is_gradio_live:
        gr.HTML("<div class='bg-wrapper'>")
    else:
        gr.HTML("<div>")

    # Floating header
    gr.HTML("""
    <div class="header">
        <div class="brand">
            <img src="/assets/odisha_icon.png" alt="logo" class="logo"/>
            <div>
                <div class="title">ଗ୍ରାମଏଆଇ <span class="subtitle-en">GramAI</span></div>
                <div class="tag">Your Doctor + Kisan Mitra — Odisha</div>
            </div>
        </div>
        <div class="header-actions">
            <button class="theme-toggle" id="theme-toggle">Light</button>
        </div>
    </div>
    """)

    # Main card
    gr.HTML("<div class='card'>")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["Paddy / ଧାନ", "Skin / ଚର୍ମ"], value="Paddy / ଧାନ", label="Mode")
            img_input = gr.Image(type="pil", label="Upload a clear photo (leaf / skin)", height=380)
            gr.Markdown("**Tips:** Use sunlight, show full lesion/leaf, avoid blur.")
        with gr.Column(scale=1):
            # Analyze button with spinner
            btn = gr.Button(value="Analyze • ଚିହ୍ନଟ କରନ୍ତୁ", elem_id="analyze-btn")
            spinner = gr.HTML("<div id='analyze-spinner' style='display:none'>🔎 Analyzing…</div>")
            conf = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ", lines=8)
            eng_out = gr.Textbox(label="English", lines=8)
            audio = gr.Audio(label="Odia audio")
    gr.HTML("</div>")  # end card

    # footer
    gr.HTML("""
    <div class='footer'>
        <div>Made with ❤️ for Odisha Farmers • Northern Region 2025</div>
        <div class='credits'>Team GramAI</div>
    </div>
    """)

    # if wrapped, close wrapper
    gr.HTML("</div>")

    # JS to show spinner while prediction runs and disable button
    demo.load(js="""
    () => {
        const btn = document.getElementById('analyze-btn');
        const spinner = document.getElementById('analyze-spinner');
        if (btn) {
            btn.addEventListener('click', () => {
                if (spinner) spinner.style.display = 'block';
            });
        }
    }
    """)

    # Wire button to prediction; spinner handled via simple JS above
    btn.click(predict, [img_input, mode], [conf, odia_out, eng_out, audio])

demo.launch(share=True)
