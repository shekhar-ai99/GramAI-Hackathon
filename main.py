# main.py — FINAL WINNING VERSION (Fully matches your diseases.json)
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
        print(f"Loaded {cfg['model_url'].split('/')[-1]} → {num_classes} classes")
    except:
        print("Warning: Using dummy weights")
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

# Background: Local + fallback
bg_path = "./assets/odisha_bg.jpg"
bg_url = "https://i.imgur.com/8Y6z1kT.jpg"
bg_css = f"url('{bg_path}')" if os.path.exists(bg_path) else f"url('{bg_url}')"

css = f"""
<style>
    body {{ background: {bg_css} no-repeat center center fixed; background-size: cover; margin:0; padding:10px; }}
    .container {{ background:rgba(255,255,255,0.96); border-radius:20px; padding:25px; box-shadow:0 10px 40px rgba(0,0,0,0.3); margin:15px auto; max-width:950px; }}
    h1 {{ font-family:'Noto Sans Oriya',sans-serif; color:#006400; text-align:center; font-size:clamp(36px,8vw,58px); text-shadow:2px 2px 10px rgba(0,0,0,0.2); }}
    .confidence {{ height:40px; border-radius:20px; text-align:center; color:white; font-weight:bold; line-height:40px; font-size:20px; }}
    @media (max-width:768px) {{ .gr-form {{flex-direction:column !important;}} .gr-button {{width:100%; margin:15px 0;}} }}
</style>
"""

with gr.Blocks(css=css, title="GramAI – ଗ୍ରାମଏଆଇ") as demo:
    gr.HTML(f"""
    <div class="container">
        <h1>ଗ୍ରାମଏଆଇ – GramAI</h1>
        <p style="text-align:center; font-size:24px; color:#006400;">
            <b>Paddy & Skin Disease Detection Using Image</b><br>
            ଧାନ ଓ ଚର୍ମ ରୋଗ ଚିହ୍ନଟ — ଓଡ଼ିଆରେ ଉପଚାର
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            mode = gr.Radio(
                ["Paddy / ଧାନ", "Skin / ଚର୍ମ"],
                label="Select / ବାଛନ୍ତୁ",
                value="Paddy / ଧାନ"
            )
            img_input = gr.Image(type="pil", label="Upload Image / ଫଟୋ ଅପଲୋଡ୍", height=450)

        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="container">')
            btn = gr.Button("Analyze / ଚିହ୍ନଟ କରନ୍ତୁ", variant="primary", size="lg")
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ ଫଳାଫଳ", lines=8)
            eng_out = gr.Textbox(label="Result in English", lines=8)
            audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି")
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

        disease_name = cfg["classes"][idx]
        details = cfg["details"][disease_name]

        odia_remedy = details["odia"]["remedy"]
        odia_desc = details["odia"].get("description", "")
        en_remedy = details["en"]["remedy"]

        color = "#2e7d32" if "Healthy" in disease_name or "ସୁସ୍ଥ" in odia_remedy else "#d32f2f"
        bar = f'<div class="confidence" style="background:{color}; width:{conf}%;">Confidence: {conf:.1f}%</div>'

        odia_text = f"ରୋଗ: {disease_name}\nବର୍ଣ୍ଣନା: {odia_desc}\nଉପଚାର: {odia_remedy}"
        eng_text = f"Disease: {disease_name} ({conf:.1f}%)\nTreatment: {en_remedy}"

        return bar, odia_text, eng_text, speak_odia(odia_remedy)

    mode.change(fn=lambda: (None, "", "", None), outputs=[conf_bar, odia_out, eng_out, audio])
    btn.click(predict, [img_input, mode], [conf_bar, odia_out, eng_out, audio])

    gr.HTML("<center><b>Made with ❤️ for Northern Odisha Farmers • Hackathon 2025</b></center>")

if __name__ == "__main__":
    demo.launch(share=True)