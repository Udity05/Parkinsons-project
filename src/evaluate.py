import torch
from torch.utils.data import DataLoader
from dataset import ParkinsonDataset
from model import ParkinsonModel
from config import *
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate():
    # Load dataset
    dataset = ParkinsonDataset("../data")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load trained model
    model = ParkinsonModel().to(DEVICE)
    model.load_state_dict(
        torch.load("parkinson_model.pth", map_location=DEVICE)
    )
    model.eval()

    all_preds = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print("\n📊 MODEL EVALUATION RESULTS\n")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate()
