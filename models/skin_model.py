# models/skin_model.py
import torch
from torchvision import transforms, models
from PIL import Image
import os

# ----------------------------------------------------
# 1) LOCAL → 2) GITHUB RAW → 3) HUGGINGFACE fallback
# ----------------------------------------------------
LOCAL_PATH = "models/skin_model_production.pth"
GITHUB_URL = "https://github.com/shekhar-ai99/GramAI-Hackathon/blob/main/models/skin_model_production.pth"
HUGGINGFACE_URL = "https://huggingface.co/spaces/ahmedshahriar/Skin_Disease/resolve/main/skin_model.pth"

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
        print("✓ Skin model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠ Failed to load from {path_or_url}: {e}")
        return False

# ----------------------------------------------------
# BUILD MODEL
# ----------------------------------------------------
model = models.mobilenet_v3_small(pretrained=False)

# Your skin model has 7 classes
model.classifier[3] = torch.nn.Linear(1024, 7)

# ORDER OF ATTEMPTS
if os.path.exists(LOCAL_PATH):
    ok = safe_load_state_dict(model, LOCAL_PATH)
    if not ok:
        ok = safe_load_state_dict(model, GITHUB_URL)
else:
    ok = safe_load_state_dict(model, GITHUB_URL)

if not ok:
    safe_load_state_dict(model, HUGGINGFACE_URL)

model.eval()

# ----------------------------------------------------
# CLASSES + REMEDIES
# ----------------------------------------------------
classes = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "Ringworm (Dadru)",
    "Scabies",
    "Fungal Infection",
    "Healthy Skin"
]

remedies = {
    "Acne": "ମୁହଁ ସଫା ରଖନ୍ତୁ | ବେଞ୍ଜୋଇଲ୍ ପେରଅକ୍ସାଇଡ୍ ବ୍ୟବହାର କରନ୍ତୁ",
    "Eczema": "ମୋଇଶ୍ଚରାଇଜର ଲଗାନ୍ତୁ | ସୁଗନ୍ଧ ଥିବା ସାବୁନ୍ ଏଡାନ୍ତୁ",
    "Psoriasis": "ଡାକ୍ତରଙ୍କ ପରାମର୍ଶ ନିଅନ୍ତୁ",
    "Ringworm (Dadru)": "କ୍ଲୋଟ୍ରିମାଜୋଲ୍ କ୍ରିମ୍ ଦିନକୁ 2 ଥର ଲଗାନ୍ତୁ",
    "Scabies": "ପରମେଥ୍ରିନ୍ ଲୋଶନ ଲଗାନ୍ତୁ + ସମଗ୍ର ପରିବାରକୁ ଚିକିତ୍ସା କରନ୍ତୁ",
    "Fungal Infection": "ଏଣ୍ଟି-ଫଙ୍ଗାଲ୍ କ୍ରିମ୍ ଲଗାନ୍ତୁ | ସଫା ଓ ଶୁଖିଲା ରଖନ୍ତୁ",
    "Healthy Skin": "ଆପଣଙ୍କ ଚର୍ମ ସୁସ୍ଥ ଅଛି!"
}

# ----------------------------------------------------
# PREDICT FUNCTION
# ----------------------------------------------------
def predict_skin(img: Image.Image) -> str:
    if img is None:
        return "ଫଟୋ ଦିଅନ୍ତୁ"

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = prob.argmax().item()
        conf = prob[idx].item() * 100

    label = classes[idx]
    remedy = remedies.get(label, "ଡାକ୍ତରଙ୍କୁ ଦେଖାନ୍ତୁ")

    return f"ଚିହ୍ନଟ: {label} ({conf:.1f}%)\nଉପଚାର: {remedy}"
