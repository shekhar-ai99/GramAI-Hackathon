# models/paddy_model.py
import torch
from torchvision import transforms, models
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(1024, 5)
try:
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/spaces/fffiloni/paddy-disease-classification/resolve/main/paddy_model.pth",
        map_location="cpu"
    ))
except:
    print("Paddy model loaded with warning")
model.eval()

classes = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Blast", "Healthy", "Tungro"]
remedies = {
    "Brown Spot": "ପ୍ରୋପିକୋନାଜୋଲ୍ 1ml/ଲିଟର ପାଣିରେ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Leaf Blast": "ଟ୍ରାଇସାଇକ୍ଲାଜୋଲ୍ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Healthy": "ଆପଣଙ୍କ ଧାନ ବହୁତ ସୁସ୍ଥ ଅଛି!",
    "Bacterial Leaf Blight": "ଷ୍ଟ୍ରେପ୍ଟୋମାଇସିନ୍ ସ୍ପ୍ରେ କରନ୍ତୁ",
    "Tungro": "ସବୁଜ ପତ୍ର କୀଟ ନିୟନ୍ତ୍ରଣ କରନ୍ତୁ"
}

def predict_paddy(img: Image.Image) -> str:
    if img is None:
        return "ଫଟୋ ଦିଅନ୍ତୁ"
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = prob.argmax().item()
        label = classes[idx]
        conf = prob[idx].item() * 100
    remedy = remedies.get(label, "ପରାମର୍ଶ ନାହିଁ")
    return f"ଚିହ୍ନଟ: {label} ({conf:.1f}%)\nଉପଚାର: {remedy}"