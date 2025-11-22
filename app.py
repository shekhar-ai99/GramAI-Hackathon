def predict_image(img, mode):
"""GramAI Swasthya O Krushi Sahayak – Northern Odisha AI Hackathon 2025
One app → Paddy Disease + Skin Disease + Odia Voice

This file loads two MobilenetV3 models (paddy + skin), provides a
`predict_image()` function that accepts a PIL image (or uses a sample image)
and returns Odia text + an audio file produced with gTTS.
"""

import os
import io
import traceback
import numpy as np
from PIL import Image

import gradio as gr
import torch
from torchvision import transforms, models
from gtts import gTTS


SAMPLE_IMAGE_PATH = "sample_images/sample1.jpeg"
UPLOADED_IMAGE_FULLPATH = "/mnt/data/711BAC8A-1F53-43B6-983A-0B8A51C128D4.jpeg"

print("Loading models... (first run takes 2–3 mins; weights downloaded from URLs)")

# === Paddy Model ===
paddy_model = models.mobilenet_v3_small(pretrained=False)
try:
    paddy_model.classifier[3] = torch.nn.Linear(1024, 5)
except Exception:
    # fallback if classifier structure is different
    try:
        paddy_model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 5))
    except Exception:
        pass

try:
    paddy_model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/spaces/fffiloni/paddy-disease-classification/resolve/main/paddy_model.pth",
        map_location="cpu"
    ))
except Exception:
    print("Warning: could not download paddy model weights; continuing without them")
paddy_model.eval()
paddy_classes = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Blast", "Healthy", "Tungro"]

# === Skin Model ===
skin_model = models.mobilenet_v3_small(pretrained=False)
try:
    skin_model.classifier[3] = torch.nn.Linear(1024, 7)
except Exception:
    try:
        skin_model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 7))
    except Exception:
        pass

try:
    skin_model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/spaces/ahmedshahriar/Skin_Disease/resolve/main/skin_model.pth",
        map_location="cpu"
    ))
except Exception:
    print("Warning: could not download skin model weights; continuing without them")
skin_model.eval()
skin_classes = ["Acne", "Eczema", "Psoriasis", "Ringworm (Dadru)", "Scabies", "Fungal Infection", "Healthy Skin"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Remedies in Odia ===
paddy_remedies = {
    "Bacterial Leaf Blight": "ପାଳିତ କାମ: ଆଇଁଜିଟିକ ଆଦାନ-ପରିବର୍ତ୍ତନ; ଆବଶ୍ୟକତାନୁସାରେ କର୍ଷକମାନେ ବ୍ୟବହାର କରିବେ: ମାନାଇଯାଉଥିବା ବ୍ୟାକ୍ଟେରିଆ ନିଦାନ, ସଫାଇ, ନିୟମିତ ପରେଚାରିତ କରନ୍ତୁ.",
    "Brown Spot": "ସୁକ୍ଷ୍ମ ସଫାଇ, ମୁଖ୍ୟତଃ ଓହ୍ଲାଇବା; ଲକ୍ଷଣ ହେଲେ ଫଂଗସ ନିୟନ୍ତ୍ରଣ ଔଷଧ ଦିଅନ୍ତୁ.",
    "Leaf Blast": "ପ୍ରଭାବିତ ପତ୍ରକୁ ହଟାନ୍ତୁ, ଅନୁମତି ପ୍ରାପ୍ତ ବିମୋକ୍ଷ ଫଙ୍ଗସିସାଇଡ୍ ଲାଗାନ୍ତୁ.",
    "Healthy": "ଆପଣଙ୍କ ଧାନ ସ୍ୱସ୍ଥ ଅଛି — ଅନୁରକ୍ଷଣ ଜାରି ରଖନ୍ତୁ.",
    "Tungro": "ଭେକ୍ଟର କଣ୍ଟ୍ରୋଲ୍ (ମୋଶା) ଓ ରୋଗ ପ୍ରତିରୋଧକ କାର୍ଯ୍ୟ; ଥିବାକୁ ଜରୁରୀ ହେଲେ ବିଶେଷଜ୍ଞ ସହାୟତା ନିଅନ୍ତୁ."
}

skin_remedies = {
    "Acne": "ମୁହଁ ସଫାଇ, ଓଭର-ଇଂଫେକସନ ନ ହେବା ପାଇଁ ଡାକ୍ତରଙ୍କ ସହ ଟ୍ରିଟମେଣ୍ଟ.",
    "Eczema": "ଚର୍ମକୁ ଶିଥିଲା ରଖନ୍ତୁ, ମାଇସ୍ଚରାଇଜର୍ ବ୍ୟବହାର କରନ୍ତୁ, ଆବଶ୍ୟକ ହେଲେ ଡାକ୍ତରଙ୍କ ସହ ସଲାହ.",
    "Psoriasis": "ଡାକ୍ତର ସହ ଦେଖା କରନ୍ତୁ; ସ୍ଥାନୀୟ କ୍ରିମ୍ ଓ ଓଷଧ ଆବଶ୍ୟକ.",
    "Ringworm (Dadru)": "ଫଙ୍ଗସ୍ ରୋଗ — ସ୍ଥାନୀୟ ଏଣ୍ଟି-ଫଙ୍ଗାଲ୍ କ୍ରିମ୍/ଲୋସନ୍ ଲାଗାନ୍ତୁ; ସଫାଇ ରଖନ୍ତୁ.",
    "Scabies": "ସ୍କାବିଜ୍ ହେଲେ ଡାକ୍ତରଙ୍କ ସହ ତଦନ୍ତ; ନିର୍ଦ୍ଦିଷ୍ଟ ମେଡିକେସନ୍ ଦରକାର.",
    "Fungal Infection": "ଫଙ୍ଗସ୍ ନିୟନ୍ତ୍ରଣ — ଲୋକାଲ୍ ଔଷଧ/କ୍ରିମ୍; ସଫାଇ ଓ ସୁକ୍ଷ୍ମ ଶରୀର.",
    "Healthy Skin": "ଚର୍ମ ସ୍ୱସ୍ଥ — ସୁସ୍ଥ ଆହାର ଓ ଖୁବ ଧଲା ସଫାଇ ରଖନ୍ତୁ."
}


def predict_image(img, mode):
    """Predicts using either the paddy or skin model.

    - `img` can be a PIL.Image or a numpy array. If None, a bundled sample image is used.
    - `mode` is the radio label from the UI; detection chooses the model.

    Returns: (text_str, audio_file_path_or_None)
    """
    try:
        # If no image provided, try the sample or uploaded full path
        if img is None:
            if os.path.exists(SAMPLE_IMAGE_PATH):
                img = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
            elif os.path.exists(UPLOADED_IMAGE_FULLPATH):
                img = Image.open(UPLOADED_IMAGE_FULLPATH).convert("RGB")
            else:
                return "କୌଣସି ଛବି ଦିଆଯାଇନି (No image provided)" , None

        if not isinstance(img, Image.Image):
            # convert numpy array to PIL
            img = Image.fromarray(np.asarray(img)).convert("RGB")

        input_tensor = transform(img).unsqueeze(0)

        if "Paddy" in mode or "ଧାନ" in mode:
            model = paddy_model
            classes = paddy_classes
            remedies = paddy_remedies
        else:
            model = skin_model
            classes = skin_classes
            remedies = skin_remedies

        with torch.no_grad():
            outputs = model(input_tensor)
            # handle models that return logits or a tuple
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            top_idx = int(np.argmax(probs))
            label = classes[top_idx]
            confidence = float(probs[top_idx])

        remedy = remedies.get(label, "ପରାମର୍ଶ ଉପଲବ୍ଧ ନାହିଁ")

        odia_text = f"ଚିହ୍ନଟ: {label} ({confidence*100:.1f}%)\nପରାମର୍ଶ: {remedy}"

        # Try to generate audio in Odia; fallback to Hindi/English
        audio_path = "result_odia.mp3"
        tts = None
        for lang in ("or", "hi", "en"):
            try:
                tts = gTTS(text=odia_text, lang=lang)
                tts.save(audio_path)
                break
            except Exception:
                tts = None

        if tts is None:
            # If tts failed, just return text and no audio
            return odia_text, None

        return odia_text, audio_path

    except Exception:
        traceback.print_exc()
        return "କିଛି ତ୍ରୁଟି ଘଟିଛି (See server logs)", None


# === Gradio Interface ===
with gr.Blocks(title="GramAI") as demo:
    gr.Markdown("# 🌾🩺 GramAI – ଗ୍ରାମଏଆଇ")
    gr.Markdown("### ଓଡ଼ିଆରେ ଧାନ + ଚର୍ମ ରୋଗ ଚିହ୍ନଟ | Northern Odisha Hackathon 2025")
    mode = gr.Radio(["🌾 Paddy / ଧାନ", "🩺 Skin / ଚର୍ମ"], label="ବାଛନ୍ତୁ | Choose:")
    img = gr.Image(type="pil", label="ଏକ ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ | Upload an image")
    btn = gr.Button("🔍 ଦେଖନ୍ତୁ | Analyze")
    out_text = gr.Textbox(label="ଉତ୍ତର | Result")
    out_audio = gr.Audio(label="ଓଡ଼ିଆ ଧ୍ୱନି | Listen")
    btn.click(predict_image, [img, mode], [out_text, out_audio])

if __name__ == "__main__":
    # ensure sample_images exists
    os.makedirs("sample_images", exist_ok=True)
    demo.launch(share=True)