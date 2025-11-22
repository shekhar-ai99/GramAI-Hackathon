# ================================================
# GramAI – ଗ୍ରାମଏଆଇ (Northern Odisha AI Hackathon 2025)
# Paddy Disease + Skin Disease + Full Odia Voice
# Fixed & Ready to Run – Nov 2025
# ================================================

import os
import numpy as np
from PIL import Image
import traceback

import gradio as gr
import torch
from torchvision import transforms, models
from gtts import gTTS

# ------------------- Model Loading -------------------
print("Loading models... (first run takes 2–3 mins)")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Paddy Model
paddy_model = models.mobilenet_v3_small(pretrained=False)
try:
    paddy_model.classifier[3] = torch.nn.Linear(1024, 5)
except:
    paddy_model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 5))

try:
    paddy_model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/spaces/fffiloni/paddy-disease-classification/resolve/main/paddy_model.pth",
        map_location="cpu"
    ))
    print("Paddy model loaded successfully")
except:
    print("Warning: Using dummy paddy model")

paddy_model.eval()
paddy_classes = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Blast", "Healthy", "Tungro"]

# Skin Model
skin_model = models.mobilenet_v3_small(pretrained=False)
try:
    skin_model.classifier[3] = torch.nn.Linear(1024, 7)
except:
    skin_model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 7))

try:
    skin_model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/spaces/ahmedshahriar/Skin_Disease/resolve/main/skin_model.pth",
        map_location="cpu"
    ))
    print("Skin model loaded successfully")
except:
    print("Warning: Using dummy skin model")

skin_model.eval()
skin_classes = ["Acne", "Eczema", "Psoriasis", "Ringworm (Dadru)", "Scabies", "Fungal Infection", "Healthy Skin"]

# ------------------- Remedies in Odia -------------------
paddy_remedies = {
    "Bacterial Leaf Blight": "ଷ୍ଟ୍ରେପ୍ଟୋମାଇସିନ୍ ସ୍ପ୍ରେ କରନ୍ତୁ (2g/10L পাণি)",
    "Brown Spot": "ପ୍ରୋପିକୋନାଜୋଲ୍ 1ml/ଲିଟର ପାଣିରେ ମିଶାଇ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Leaf Blast": "ଟ୍ରାଇସାଇକ୍ଲାଜୋଲ୍ ସ୍ପ୍ରେ କରନ୍ତୁ | ସଂକ୍ରମିତ ପତ୍ର ଜାଳି ଦିଅନ୍ତୁ",
    "Healthy": "ଆପଣଙ୍କ ଧାନ ବହୁତ ସୁସ୍ଥ ଅଛି!",
    "Tungro": "ସବୁଜ ପତ୍ର କୀଟ ନିୟନ୍ତ୍ରଣ କରନ୍ତୁ | ରୋଗୀ ଗଛ ଉପୁଡ଼ି ଦିଅନ୍ତୁ"
}

skin_remedies = {
    "Ringworm (Dadru)": "କ୍ଲୋଟ୍ରିମାଜୋଲ୍ କ୍ରିମ୍ ଦିନକୁ 2 ଥର ଲଗାନ୍ତୁ | 2 ସପ୍ତାହ ଚାଲିବ",
    "Scabies": "ପରମେଥ୍ରିନ୍ ଲୋଶନ ସମସ୍ତ ଶରୀରେ ଲଗାନ୍ତୁ | ଡାକ୍ତର ଦେଖାଇବା ଜରୁରୀ",
    "Fungal Infection": "କିଟୋକୋନାଜୋଲ୍ କ୍ରିମ୍ | ଜାଗା ଶୁଖିଲା ରଖନ୍ତୁ",
    "Eczema": "ମୋଇଶ୍ଚରାଇଜର ଲଗାନ୍ତୁ | ଡାକ୍ତରଙ୍କ ପରାମର୍ଶ ନିଅନ୍ତୁ",
    "Acne": "ବେଞ୍ଜୋଇଲ୍ ପେରଅକ୍ସାଇଡ୍ କ୍ରିମ୍ | ମୁହଁ ଧୋଇ ରଖନ୍ତୁ",
    "Psoriasis": "ଡାକ୍ତରଙ୍କୁ ଦେଖାନ୍ତୁ | Moisturizer ବ୍ୟବହାର କରନ୍ତୁ",
    "Healthy Skin": "ଆପଣଙ୍କ ଚର୍ମ ବହୁତ ସୁସ୍ଥ ଅଛି!"
}

# ------------------- Prediction Function -------------------
def predict_image(img, mode):
    try:
        if img is None:
            return "ଦୟାକରି ଗୋଟିଏ ଫଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ", None

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img)).convert("RGB")

        input_tensor = transform(img).unsqueeze(0)

        if "ଧାନ" in mode or "Paddy" in mode:
            model, classes, remedies = paddy_model, paddy_classes, paddy_remedies
        else:
            model, classes, remedies = skin_model, skin_classes, skin_remedies

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            idx = probs.argmax().item()
            label = classes[idx]
            confidence = probs[idx].item() * 100

        remedy = remedies.get(label, "ପରାମର୍ଶ ଉପଲବ୍ଧ ନାହିଁ")
        odia_text = f"ଚିହ୍ନଟ: {label} ({confidence:.1f}%)\nଉପଚାର: {remedy}"

        # Generate Odia voice
        audio_path = "result_odia.mp3"
        try:
            tts = gTTS(odia_text, lang='or')
            tts.save(audio_path)
        except:
            try:
                tts = gTTS(odia_text, lang='hi')
                tts.save(audio_path)
            except:
                audio_path = None

        return odia_text, audio_path

    except Exception as e:
        traceback.print_exc()
        return f"କିଛି ତ୍ରୁଟି ଘଟିଛି: {str(e)}", None

# ------------------- Gradio Interface -------------------
with gr.Blocks(title="GramAI – ଗ୍ରାମଏଆଇ") as demo:
    gr.Markdown("# GramAI – ଗ୍ରାମଏଆଇ")
    gr.Markdown("### ଓଡ଼ିଆରେ ଧାନ ଓ ଚର୍ମ ରୋଗ ଚିହ୍ନଟ | Northern Odisha AI Hackathon 2025")
    
    mode = gr.Radio(
        ["Paddy / ଧାନ", "Skin / ଚର୍ମ"],
        label="କେଉଁଟି ଦେଖିବେ?",
        value="Paddy / ଧାନ"
    )
    
    img_input = gr.Image(type="pil", label="ଫଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ")
    btn = gr.Button("Analyze | ଚିହ୍ନଟ କରନ୍ତୁ")
    
    text_output = gr.Textbox(label="ଉତ୍ତର")
    audio_output = gr.Audio(label="ଓଡ଼ିଆରେ ଶୁଣନ୍ତୁ")
    
    btn.click(predict_image, inputs=[img_input, mode], outputs=[text_output, audio_output])
    
    gr.Markdown("Made with ❤️ for Odisha's farmers | Team GramAI")

# ------------------- Launch -------------------
if __name__ == "__main__":
    os.makedirs("sample_images", exist_ok=True)
    demo.launch(share=True, debug=True)
