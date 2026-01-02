import torch
from model import ParkinsonModel
from preprocess import extract_mfcc
from config import DEVICE

def predict(audio_path):
    model = ParkinsonModel().to(DEVICE)
    model.load_state_dict(torch.load("parkinson_model.pth", map_location=DEVICE))
    model.eval()

    mfcc = extract_mfcc(audio_path)
    x = torch.tensor(mfcc).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()

    label = "Parkinson Detected" if pred == 1 else "Healthy"
    return label, confidence

if __name__ == "__main__":
    audio = r"C:\Users\udity\OneDrive\Desktop\Parkinsons-project\data\28 People with Parkinson's disease\6-10\Saverio S\VA2ssacvhei61M1606161744.wav"
    result, conf = predict(audio)
    print(f"Prediction: {result}")
    print(f"Confidence: {conf*100:.2f}%")
