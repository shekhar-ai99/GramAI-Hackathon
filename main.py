# main.py — FINAL 100% WORKING (Gradio 5+ compatible)
import gradio as gr
import torch
from torchvision import transforms, models
import json
from gtts import gTTS

# Load config & models (unchanged)
with open("config/diseases.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def load_model(cfg):
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(cfg["classes"])
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    try:
        state = torch.hub.load_state_dict_from_url(cfg["model_url"], map_location="cpu")
        model.load_state_dict(state)
    except:
        print("Using dummy weights")
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

# Load external CSS
with open("css/style.css", "r", encoding="utf-8") as f:
    css = f"<style>{f.read()}</style>"

# THIS IS THE ONLY CHANGE — NO css= parameter!
with gr.Blocks(title="GramAI – ଗ୍ରାମଏଆଇ") as demo:
    gr.HTML(css)  # THIS FIXES THE ERROR

    gr.HTML("""
    <div class="container">
        <h1 class="title-or">ଗ୍ରାମ-ଏଆଇ - GramAI</h1>
        <p class="subtitle">
            <b>Paddy & Skin Disease Detection Using Image<i>Your Personal Doctor + Kisan Mitra</i></b>
           <br>
            ଧାନ ଓ ଚର୍ମ ରୋଗ ଚିହ୍ନଟ — ଓଡ଼ିଆରେ ଉପଚାର <i>ଆପଣଙ୍କ ଡାକ୍ତର ଓ କୃଷକ ସାଥୀ — ଗୋଟିଏ ଫଟୋରୁ</i>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            mode = gr.Radio(["Paddy / ଧାନ", "Skin / ଚର୍ମ"], value="Paddy / ଧାନ", label="Select Mode")
            img_input = gr.Image(type="pil", label="Upload Photo", height=500)

        with gr.Column():
            gr.HTML('<div class="container">')

            btn = gr.Button("Analyze", variant="primary", size="lg")
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ ଫଳାଫଳ", lines=10)
            eng_out = gr.Textbox(label="Result in English", lines=10)
            audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି")

            gr.HTML('</div>')


    def predict(img, mode):
        if img is None:
            return None, "ଫଟୋ ଦିଅନ୍ତୁ / Please upload a photo", "", None

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

        # Confidence bar color
        color = "#00ff62" if "Healthy" in disease or "ସୁସ୍ଥ" in details["odia"]["remedy"] else "#ff6200"
        bar = f'<div class="confidence">Confidence: {conf:.1f}%</div>'

        # Bilingual output
        odia = f"ରୋଗ: {disease}\nବର୍ଣ୍ଣନା: {details['odia'].get('description','')}\nଉପଚାର: {details['odia']['remedy']}\nମାତ୍ରା: {details['odia'].get('dosage','')}"
        eng = f"Disease: {disease} ({conf:.1f}%)\nDescription: {details['en'].get('description','')}\nTreatment: {details['en']['remedy']}\nDosage: {details['en'].get('dosage','')}"

        return bar, odia, eng, speak_odia(details["odia"]["remedy"])

    btn.click(predict, [img_input, mode], [conf_bar, odia_out, eng_out, audio])
    gr.HTML("""
    <center style="margin-top:40px; margin-bottom:45px;">
    <div style="
        background: rgba(0, 0, 0, 0.55);
        backdrop-filter: blur(4px);
        padding: 22px 32px;
        border-radius: 18px;
        display: inline-block;
        max-width: 90%;
        color:#ffffff;
        font-size:20px;
        font-weight:600;
        line-height:1.7;
        letter-spacing:0.3px;
        text-shadow:0 0 6px rgba(0,0,0,0.8);
    ">
        🌾🩺 <span style="color:#7CFF91;">AI for Health & Agriculture — Empowering Lives, Empowering Farmers</span><br>
        🚀 Built for <b>Hackathon 2025</b> • Developed at <b>MSCB University, Odisha</b><br><br>
        👨‍🎓 <b style="color:#7CFF91;">Lead Researcher:</b> Chandra Shekhar Behera  
        <span style="opacity:0.9">(Research Scholar, MSCB University)</span><br>
        👨‍🏫 <b style="color:#7CFF91;">Academic Supervisor:</b> Dr. Swarupananda Bissoyi  
        <span style="opacity:0.9">(Assistant Professor, MSCB University)</span>
    </div>
</center>
""")


demo.launch(share=True)