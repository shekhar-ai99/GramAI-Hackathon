import torch
from torchvision import transforms, models
from PIL import Image
import os
import requests

# ----------------------------------------------------
# 1) LOCAL -> 2) GITHUB RAW -> 3) HUGGINGFACE fallback
# ----------------------------------------------------
LOCAL_PATH = "models/paddy_model.pth"
GITHUB_URL = "https://github.com/shekhar-ai99/GramAI-Hackathon/blob/main/models/paddy_model.pth"
HUGGINGFACE_URL = "https://huggingface.co/spaces/fffiloni/paddy-disease-classification/resolve/main/paddy_model.pth"

# ----------------------------------------------------
# TRANSFORM
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
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
# BUILD MODEL
# ----------------------------------------------------
model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(1024, 5)

# ORDER OF ATTEMPTS
if os.path.exists(LOCAL_PATH):
    ok = safe_load_state_dict(model, LOCAL_PATH)
    if not ok:
        ok = safe_load_state_dict(model, GITHUB_URL)
else:
    ok = safe_load_state_dict(model, GITHUB_URL)

if not ok:  # final fallback
    safe_load_state_dict(model, HUGGINGFACE_URL)

model.eval()

# ----------------------------------------------------
# CLASSES + REMEDIES
# ----------------------------------------------------
classes = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Blast",
    "Healthy",
    "Tungro"
]

remedies = {
    "Brown Spot": "ପ୍ରୋପିକୋନାଜୋଲ୍ 1ml/ଲିଟର ପାଣିରେ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Leaf Blast": "ଟ୍ରାଇସାଇକ୍ଲାଜୋଲ୍ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Healthy": "ଆପଣଙ୍କ ଧାନ ବହୁତ ସୁସ୍ଥ ଅଛି!",
    "Bacterial Leaf Blight": "ଷ୍ଟ୍ରେପ୍ଟୋମାଇସିନ୍ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Tungro": "ସବୁଜ ପତ୍ର କୀଟ ନିୟନ୍ତ୍ରଣ କରନ୍ତୁ"
}

# ----------------------------------------------------
# PREDICT FUNCTION
# ----------------------------------------------------
def predict_paddy(img: Image.Image) -> str:
    if img is None:
        return "ଫଟୋ ଦିଅନ୍ତୁ"

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = prob.argmax().item()
        conf = prob[idx].item() * 100

    label = classes[idx]
    remedy = remedies.get(label, "ପରାମର୍ଶ ନାହିଁ")

    return f"ଚିହ୍ନଟ: {label} ({conf:.1f}%)\nଉପଚାର: {remedy}"
