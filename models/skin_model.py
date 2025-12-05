# models/skin_model.py
import torch
from torchvision import transforms, models
from PIL import Image
import os

# ----------------------------------------------------
# 1) LOCAL → 2) GITHUB RAW → 3) HUGGINGFACE fallback
# ----------------------------------------------------
LOCAL_PATH = "models/skin_model_1.pth"

GITHUB_URL = "https://raw.githubusercontent.com/shekhar-ai99/GramAI-Hackathon/main/models/skin_model_1.pth"
HUGGINGFACE_URL = "https://huggingface.co/spaces/ahmedshahriar/Skin_Disease/resolve/main/skin_model.pth"


# ----------------------------------------------------
# TRANSFORMS (MUST MATCH TRAINING)
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
        print("✓ Skin model loaded successfully")
        return True

    except Exception as e:
        print(f"⚠ Failed to load {path_or_url}: {e}")
        return False


# ----------------------------------------------------
# BUILD MODEL - MobileNetV3 Small (7 classes)
# ----------------------------------------------------
def build_model():
    model = models.mobilenet_v3_small(weights=None)   # SAME AS TRAINING
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 7)  # YOU HAVE 7 SKIN CLASSES
    return model

model = build_model()
model.eval()




# ----------------------------------------------------
# LOAD WEIGHTS (Local → GitHub → HF)
# ----------------------------------------------------
loaded = False

if os.path.exists(LOCAL_PATH):
    loaded = safe_load_state_dict(model, LOCAL_PATH)

if not loaded:
    loaded = safe_load_state_dict(model, GITHUB_URL)

if not loaded:
    loaded = safe_load_state_dict(model, HUGGINGFACE_URL)

if not loaded:
    raise RuntimeError("❌ Could not load any skin model checkpoint")


# ----------------------------------------------------
# CORRECT CLASS ORDER — MUST MATCH TRAINING FOLDERS
# ----------------------------------------------------
classes = [
    "Acne",
    "Benign Growth",
    "Eczema",
    "Fungal Nail",
    "Psoriasis",
    "Ringworm",
    "Scabies"
]


# ----------------------------------------------------
# REMEDIES (matching corrected class names)
# ----------------------------------------------------
remedies = {
    "Acne": "ମୁହଁ ସଫା ରଖନ୍ତୁ | ବେଞ୍ଜୋଇଲ୍ ପେରଅକ୍ସାଇଡ୍ ବ୍ୟବହାର କରନ୍ତୁ",
    "Benign Growth": "ସୁମଧୁର ଗଠାନ୍ | ଡାକ୍ତରଙ୍କୁ ଯାଞ୍ଚ କରନ୍ତୁ",
    "Eczema": "ମୋଇଶ୍ଚରାଇଜର ଲଗାନ୍ତୁ | ସୁଗନ୍ଧିତ ସାବୁନ୍ ଏଡାନ୍ତୁ",
    "Fungal Nail": "ଏଣ୍ଟି-ଫଙ୍ଗାଲ୍ କ୍ରିମ୍ ବ୍ୟବହାର କରନ୍ତୁ",
    "Psoriasis": "ଡାକ୍ତରଙ୍କ ପରାମର୍ଶ ନିଅନ୍ତୁ | ବିଶେଷ ଔଷଧ ଦରକାର",
    "Ringworm": "କ୍ଲୋଟ୍ରିମାଜୋଲ୍ କ୍ରିମ୍ ଦିନକୁ 2ଥର ଲଗାନ୍ତୁ",
    "Scabies": "ପରମେଥ୍ରିନ୍ ଲୋଶନ ବ୍ୟବହାର କରନ୍ତୁ | ପରିବାର ସଭ୍ୟଙ୍କୁ ମଧ୍ୟ ଚିକିତ୍ସା କରନ୍ତୁ"
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
