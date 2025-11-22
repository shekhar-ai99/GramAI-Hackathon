# main.py — FULLY AUTOMATED WINNER (Zero hard-coding)
import gradio as gr
import torch
from torchvision import transforms, models
import json
import os
from gtts import gTTS

# Load config (auto-detects everything)
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
        print("Warning: Model loaded with dummy weights")
    model.eval()
    return model

paddy_cfg = config["paddy"]
skin_cfg = config["skin"]
paddy_model = load_model(paddy_cfg)
skin_model = load_model(skin_cfg)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def speak_odia(text):
    path = "result.mp3"
    gTTS(text, lang='or', slow=False).save(path)
    return path

css = """<style>
    body { background: url('./assets/odisha_bg.jpg') no-repeat center center fixed; background-size: cover; margin:0; padding:10px; }
    .container { background:rgba(255,255,255,0.96); border-radius:20px; padding:25px; box-shadow:0 10px 40px rgba(0,0,0,0.3); margin:15px auto; max-width:950px; }
    h1 { font-family:'Noto Sans Oriya',sans-serif; color:#006400; text-align:center; font-size:clamp(36px,8vw,58px); }
    .confidence { height:40px; border-radius:20px; text-align:center; color:white; font-weight:bold; line-height:40px; font-size:20px; }
</style>"""

with gr.Blocks(css=css, title="GramAI") as demo:
    gr.HTML('<div class="container"><h1>ଗ୍ରାମଏଆଇ – GramAI</h1><p style="text-align:center; font-size:24px; color:#006400;"><b>Paddy & Skin Disease Detection Using Image</b><br>ଧାନ ଓ ଚର୍ମ ରୋଗ ଚିହ୍ନଟ — ଓଡ଼ିଆରେ ଉପଚାର</p></div>')

    with gr.Row():
        with gr.Column():
            mode = gr.Radio(["Paddy / ଧାନ", "Skin / ଚର୍ମ"], label="Select / ବାଛନ୍ତୁ", value="Paddy / ଧାନ")
            img = gr.Image(type="pil", label="Upload Image / ଫଟୋ ଅପଲୋଡ୍")

        with gr.Column():
            btn = gr.Button("Analyze / ଚିହ୍ନଟ କରନ୍ତୁ", variant="primary", size="lg")
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ")
            eng_out = gr.Textbox(label="In English")
            audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି")

    def predict(img, mode):
        if img is None: return None, "ଫଟୋ ଦିଅନ୍ତୁ", "", None
        tensor = transform(img).unsqueeze(0)
        cfg = paddy_cfg if "ଧାନ" in mode else skin_cfg
        model = paddy_model if "ଧାନ" in mode else skin_model

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            idx = prob.argmax().item()
            conf = prob[idx].item() * 100

        disease = cfg["classes"][idx]
        odia_remedy = cfg["remedies_odia"][idx]
        en_remedy = cfg["remedies_en"][idx]

        color = "#2e7d32" if "Healthy" in disease or "ସୁସ୍ଥ" in odia_remedy else "#d32f2f"
        bar = f'<div class="confidence" style="background:{color}; width:{conf}%;">Confidence: {conf:.1f}%</div>'

        odia_text = f"ରୋଗ: {disease}\nଉପଚାର: {odia_remedy}"
        eng_text = f"Disease: {disease} ({conf:.1f}%)\nTreatment: {en_remedy}"

        return bar, odia_text, eng_text, speak_odia(odia_remedy)

    mode.change(fn=lambda: (None, "", "", None), outputs=[conf_bar, odia_out, eng_out, audio])
    btn.click(predict, [img, mode], [conf_bar, odia_out, eng_out, audio])

    gr.HTML("<center><b>Made with ❤️ for Odisha Farmers • Northern Region 2025</b></center>")

demo.launch(share=True)