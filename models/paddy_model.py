import torch
from torchvision import transforms, models
from PIL import Image
import os
import requests

# ----------------------------------------------------
# 1) LOCAL -> 2) GITHUB -> 3) HUGGINGFACE fallback
# ----------------------------------------------------
LOCAL_PATH = "models/paddy_model_1.pth"
GITHUB_URL = "https://github.com/shekhar-ai99/GramAI-Hackathon/raw/main/models/paddy_model_1.pth"
HUGGINGFACE_URL = "https://huggingface.co/spaces/fffiloni/paddy-disease-classification/resolve/main/paddy_model.pth"

# ----------------------------------------------------
# TRANSFORM (MUST MATCH TRAINING)
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------------------
# SAFE LOADER
# ----------------------------------------------------
def safe_load_state_dict(model, path_or_url):
    try:
        if path_or_url.startswith("http"):
            print(f"Downloading: {path_or_url}")
            state = torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu")
        else:
            print(f"Loading local file: {path_or_url}")
            state = torch.load(path_or_url, map_location="cpu")

        model.load_state_dict(state)
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠ Failed to load from {path_or_url}: {e}")
        return False

# ----------------------------------------------------
# BUILD MODEL (MUST MATCH TRAINED MODEL)
# MobileNetV3-Small → classifier output = 4 classes
# ----------------------------------------------------
# def build_model():
#     model = models.mobilenet_v3_small(weights=None)  # MATCH TRAINING EXACTLY
#     in_features = model.classifier[3].in_features
#     model.classifier[3] = torch.nn.Linear(in_features, 4)
#     return model


def build_model():
    model = models.mobilenet_v3_small(weights=None)  # MUST MATCH TRAINING
    model.classifier[0] = torch.nn.Linear(576, 1024)
    model.classifier[3] = torch.nn.Linear(1024, 4)
    return model

model = build_model()
model.eval()



# ----------------------------------------------------
# LOAD CHECKPOINT (Local → GitHub → HuggingFace)
# ----------------------------------------------------
loaded = False

if os.path.exists(LOCAL_PATH):
    loaded = safe_load_state_dict(model, LOCAL_PATH)

if not loaded:
    loaded = safe_load_state_dict(model, GITHUB_URL)

if not loaded:
    loaded = safe_load_state_dict(model, HUGGINGFACE_URL)

if not loaded:
    raise RuntimeError("❌ Could not load any model checkpoint")


# ----------------------------------------------------
# CLASS LABELS (MUST MATCH TRAINING ORDER)
# ----------------------------------------------------
classes = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro"
]

# ----------------------------------------------------
# Remedies (Your Odia text preserved)
# ----------------------------------------------------
remedies = {
    "Bacterialblight": "ଷ୍ଟ୍ରେପ୍ଟୋମାଇସିନ୍ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Blast": "ଟ୍ରାଇସାଇକ୍ଲାଜୋଲ୍ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Brownspot": "ପ୍ରୋପିକୋନାଜୋଲ୍ 1ml/ଲିଟର ପାଣିରେ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Tungro": "ସବୁଜ ପତ୍ର କୀଟ ନିୟନ୍ତ୍ରଣ କରନ୍ତୁ"
}

# ----------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------
def predict_paddy(img: Image.Image) -> str:
    if img is None:
        return ""   # ← EMPTY STRING clears the output box

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = prob.argmax().item()
        conf = prob[idx].item() * 100

    label = classes[idx]
    remedy = remedies.get(label, "ପରାମର୍ଶ ନାହିଁ")

    return f"ଚିହ୍ନଟ: {label} ({conf:.1f}%)\nଉପଚାର: {remedy}"
