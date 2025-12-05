import gradio as gr
import torch
from torchvision import transforms, models
import json
from gtts import gTTS
from gtts.lang import tts_langs

# Patch gTTS to recognize Odia (Oriya)
if "or" not in tts_langs():
    tts_langs()["or"] = "Odia"

# Load config
with open("config/diseases.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def load_model(cfg):
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(cfg["classes"])
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    try:
        print("Downloading from:", cfg["model_url"])
        state = torch.hub.load_state_dict_from_url(cfg["model_url"], map_location="cpu")
    except Exception as e:
        print("⚠️ Download failed:", e)
        print("➡️ Loading from local:", cfg["local_path"])
        state = torch.load(cfg["local_path"], map_location="cpu")


    model.load_state_dict(state)
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
        gTTS(text, lang="or", slow=False).save(path)
    except Exception as e:
        print("Odia TTS failed:", e)
        gTTS(text, lang="en", slow=False).save(path)
    return path


def clear_outputs():
    return "", "", "", None

def predict(img, mode):
    if img is None:
        return "", "ଫଟୋ ଦିଅନ୍ତୁ / Please upload a photo", "", None

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

    bar = f"<div class='confidence'>Confidence: {conf:.1f}%</div>"
    odia = f"ରୋଗ: {disease}\nବର୍ଣ୍ଣନା: {details['odia'].get('description','')}\nଉପଚାର: {details['odia']['remedy']}\nମାତ୍ରା: {details['odia'].get('dosage','')}"
    eng = f"Disease: {disease} ({conf:.1f}%)\nDescription: {details['en'].get('description','')}\nTreatment: {details['en']['remedy']}\nDosage: {details['en'].get('dosage','')}"

    return bar, odia, eng, speak_odia(details["odia"]["remedy"])


# Load CSS
with open("css/style.css", "r", encoding="utf-8") as f:
    css = f"<style>{f.read()}</style>"

with gr.Blocks(title="GramAI – ଗ୍ରାମଏଆଇ") as demo:

    gr.HTML(css)

    with gr.Row():
        with gr.Column():
            mode = gr.Radio(["Paddy / ଧାନ", "Skin / ଚର୍ମ"], value="Paddy / ଧାନ")
            img_input = gr.Image(type="pil", label="Upload Photo", height=500)

        with gr.Column():
            btn = gr.Button("Analyze", variant="primary")
            conf_bar = gr.HTML()
            odia_out = gr.Textbox(label="ଓଡ଼ିଆରେ ଫଳାଫଳ")
            eng_out = gr.Textbox(label="Result in English")
            audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି")

    # Clear outputs when mode changes
    mode.change(
        lambda _: (None, "", "", "", None),
        inputs=None,
        outputs=[img_input, conf_bar, odia_out, eng_out, audio]
    )

    # Clear only outputs when new image is uploaded
    img_input.change(
        clear_outputs,
        inputs=None,
        outputs=[conf_bar, odia_out, eng_out, audio]
    )

    # Run prediction
    btn.click(
    predict,
    inputs=[img_input, mode],
    outputs=[conf_bar, odia_out, eng_out, audio]
)





#     gr.HTML("""
#     <center style="margin-top:40px; margin-bottom:45px;">
#     <div style="
#         background: rgba(0, 0, 0, 0.55);
#         backdrop-filter: blur(4px);
#         padding: 22px 32px;
#         border-radius: 18px;
#         display: inline-block;
#         max-width: 90%;
#         color:#ffffff;
#         font-size:20px;
#         font-weight:600;
#         line-height:1.7;
#         letter-spacing:0.3px;
#         text-shadow:0 0 6px rgba(0,0,0,0.8);overflow: visible;
#     ">
#         🌾🩺 <span style="color:#7CFF91;">AI for Health & Agriculture — Empowering Lives, Empowering Farmers</span><br>
#         🚀 Built for <b>Hackathon 2025</b> • Developed at <b>MSCB University, Odisha</b><br><br>
#         👨‍🎓 <b style="color:#7CFF91;">Lead Researcher:</b> Chandra Shekhar Behera  
#         <span style="opacity:0.9">(Research Scholar, MSCB University)</span><br>
#         👨‍🏫 <b style="color:#7CFF91;">Academic Supervisor:</b> Dr. Swarupananda Bissoyi  
#         <span style="opacity:0.9">(Assistant Professor, MSCB University)</span>
#     </div>
# </center>
# """)
gr.HTML("""
<div style="
    width: 100%;
    padding: 30px 10px;
    margin-top: 40px;
    margin-bottom: 40px;
    background: rgba(0, 0, 0, 0.60);
    backdrop-filter: blur(6px);
    border-radius: 12px;
    color: #ffffff;
    text-align: center;
    box-sizing: border-box;
">

    <h3 style="margin: 0; padding: 0;">
        🌾🩺 <span style="color:#7CFF91;">AI for Health & Agriculture</span>
    </h3>

    <p style="
        font-size: 18px;
        margin-top: 12px;
        margin-bottom: 18px;
        line-height: 1.6;
    ">
        Empowering Lives, Empowering Farmers  
        <br>
        🚀 Built for <b>Hackathon 2025</b> • Developed at <b>MSCB University, Odisha</b>
    </p>

    <p style="
        font-size: 16px;
        margin: 0;
        opacity: 0.9;
        line-height: 1.6;
    ">
        👨‍🎓 <b>Lead Researcher:</b> Chandra Shekhar Behera  
        (Research Scholar, MSCB University)
        <br>
        👨‍🏫 <b>Academic Supervisor:</b> Dr. Swarupananda Bissoyi  
        (Assistant Professor, MSCB University)
    </p>

</div>
""")


demo.launch(share=True)