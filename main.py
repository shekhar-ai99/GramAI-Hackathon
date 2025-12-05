import gradio as gr
import torch
from torchvision import transforms, models
import json
from gtts import gTTS
from gtts.lang import tts_langs
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ---------------------------------------------------------
# PATCH: Add Odia to gTTS if missing
# ---------------------------------------------------------
if "or" not in tts_langs():
    tts_langs()["or"] = "Odia"

# ---------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------
with open("config/diseases.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# ---------------------------------------------------------
# LOAD DISEASE MODELS
# ---------------------------------------------------------
def load_model(cfg):
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(cfg["classes"])
    model.classifier[3] = torch.nn.Linear(1024, num_classes)

    try:
        print("Downloading from:", cfg["model_url"])
        state = torch.hub.load_state_dict_from_url(cfg["model_url"], map_location="cpu")
    except Exception as e:
        print("⚠ Download failed:", e)
        print("➡ Loading local:", cfg["local_path"])
        state = torch.load(cfg["local_path"], map_location="cpu")

    model.load_state_dict(state)
    model.eval()
    return model

paddy_model = load_model(config["paddy"])
skin_model = load_model(config["skin"])

# ---------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# EFFICIENTNET IMAGE-TYPE DETECTOR (PLANT vs SKIN)
# ---------------------------------------------------------
type_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
type_model.eval()

imagenet_labels = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]

type_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def detect_image_type(img):
    tensor = type_transform(img).unsqueeze(0)

    with torch.no_grad():
        out = type_model(tensor)
        probs = torch.softmax(out, 1)[0]
        idx = probs.argmax().item()
        label = imagenet_labels[idx].lower()
        conf = probs[idx].item() * 100

    return label, conf

# Keyword groups
plant_keywords = ["leaf", "plant", "tree", "rice", "vegetation", "grass"]
skin_keywords = ["skin", "person", "human", "hand", "arm", "face"]


# ---------------------------------------------------------
# TTS
# ---------------------------------------------------------
def speak_odia(text):
    path = "result_odia.mp3"
    try:
        gTTS(text, lang="or", slow=False).save(path)
    except:
        gTTS(text, lang="en").save(path)
    return path

# ---------------------------------------------------------
# CLEAR OUTPUTS
# ---------------------------------------------------------
def clear_outputs():
    return "", "", "", None

# ---------------------------------------------------------
# PREDICT FUNCTION (WITH MISMATCH DETECTION)
# ---------------------------------------------------------
THRESHOLD = 12  # Confidence margin to detect wrong-category upload

def predict(img, mode):
    if img is None:
        return "", "ଫଟୋ ଦିଅନ୍ତୁ / Please upload a photo", "", None

    tensor = transform(img).unsqueeze(0)

    # Select correct model
    if "ଧାନ" in mode:
        cfg = config["paddy"]
        model = paddy_model
    else:
        cfg = config["skin"]
        model = skin_model

    # Run the selected model
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = prob.argmax().item()
        conf = prob[idx].item() * 100

    # Extract result
    disease = cfg["classes"][idx]
    details = cfg["details"][disease]

    bar = f"<div class='confidence'>Confidence: {conf:.1f}%</div>"

    odia = (
        f"ରୋଗ: {disease}\n"
        f"ବର୍ଣ୍ଣନା: {details['odia'].get('description','')}\n"
        f"ଉପଚାର: {details['odia']['remedy']}\n"
        f"ମାତ୍ରା: {details['odia'].get('dosage','')}"
    )

    eng = (
        f"Disease: {disease} ({conf:.1f}%)\n"
        f"Description: {details['en'].get('description','')}\n"
        f"Treatment: {details['en']['remedy']}\n"
        f"Dosage: {details['en'].get('dosage','')}"
    )

    audio_path = speak_odia(details["odia"]["remedy"])

    return bar, odia, eng, audio_path



# ---------------------------------------------------------
# LOAD CSS
# ---------------------------------------------------------
with open("css/style.css", "r", encoding="utf-8") as f:
    css = f"<style>{f.read()}</style>"

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
with gr.Blocks(title="GramAI – ଗ୍ରାମଏଆଇ") as demo:

    gr.HTML(css)

    # -------------------------------------------------
    # HEADER (Restored)
    # -------------------------------------------------
    gr.HTML("""
    <div class="container">
        <h1 class="title-or">ଗ୍ରାମ-ଏଆଇ - GramAI</h1>
        <p class="subtitle">
            Paddy & Skin Disease Detection Using AI<br>
            <i>ଧାନ ଓ ଚର୍ମ ରୋଗ — ଗୋଟିଏ ଫଟୋରେ ଚିହ୍ନଟ</i>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            mode = gr.Radio(["Paddy / ଧାନ", "Skin / ଚର୍ମ"],
                            value="Paddy / ଧାନ",
                            label="Select Mode")
            btn = gr.Button("Analyze", variant="primary")
            img_input = gr.Image(type="pil", label="Upload Photo", height=500)

        with gr.Column():
            
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ ଫଳାଫଳ", lines=8)
            eng_out = gr.Textbox(label="Result in English", lines=8)
            audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି")

    # Reset Outputs
    mode.change(lambda _: (None, "", "", "", None),
                None,
                [img_input, conf_bar, odia_out, eng_out, audio])

    img_input.change(clear_outputs, None,
                     [conf_bar, odia_out, eng_out, audio])

    btn.click(predict,
              inputs=[img_input, mode],
              outputs=[conf_bar, odia_out, eng_out, audio])

    # -------------------------------------------------
    # FOOTER (Restored)
    # -------------------------------------------------
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
        text-shadow:0 0 6px rgba(0,0,0,0.8);
    ">
        🌾🩺 <span style="color:#7CFF91;">AI for Health & Agriculture</span><br>
        Built for <b>Hackathon 2025</b> • MSCB University, Odisha<br><br>
        👨‍🎓 <b style="color:#7CFF91;">Lead Researcher:</b> Chandra Shekhar Behera<br>
        👨‍🏫 <b style="color:#7CFF91;">Supervisor:</b> Dr. Swarupananda Bissoyi
    </div>
    </center>
    """)

# ---------------------------------------------------------
demo.launch(share=True)
