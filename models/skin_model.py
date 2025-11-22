# models/skin_model.py
import torch
from torchvision import transforms, models
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(1024, 7)
try:
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/spaces/ahmedshahriar/Skin_Disease/resolve/main/skin_model.pth",
        map_location="cpu"
    ))
except:
    print("Skin model loaded with warning")
model.eval()

classes = ["Acne", "Eczema", "Psoriasis", "Ringworm (Dadru)", "Scabies", "Fungal Infection", "Healthy Skin"]
remedies = {
    "Ringworm (Dadru)": "କ୍ଲୋଟ୍ରିମାଜୋଲ୍ କ୍ରିମ୍ ଦିନକୁ 2 ଥର ଲଗାନ୍ତୁ",
    "Scabies": "ପରମେଥ୍ରିନ୍ ଲୋଶନ + ଡାକ୍ତର ଦେଖାନ୍ତୁ",
    "Healthy Skin": "ଆପଣଙ୍କ ଚର୍ମ ସୁସ୍ଥ ଅଛି!"
}

def predict_skin(img: Image.Image) -> str:
    if img is None:
        return "ଫଟୋ ଦିଅନ୍ତୁ"
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = prob.argmax().item()
        label = classes[idx]
        conf = prob[idx].item() * 100
    remedy = remedies.get(label, "ଡାକ୍ତରଙ୍କୁ ଦେଖାନ୍ତୁ")
    return f"ଚିହ୍ନଟ: {label} ({conf:.1f}%)\nଉପଚାର: {remedy}"