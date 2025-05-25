import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from models.model import BertWithScalarFeatures, CommentDataset
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    comments = df['comment'].tolist()
    scalar_features = df[[
        'hate_score', 'toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit',
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive',
        'emotion_anger', 'emotion_fear', 'emotion_joy', 'emotion_love', 'emotion_sadness', 'emotion_surprise'
    ]].values.astype(np.float32)
    
    labels = df['label'].values

    _, val_comments, _, val_scalar, _, y_val = train_test_split(
        comments, scalar_features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    val_dataset = CommentDataset(
        comments=val_comments,
        scalar_features=val_scalar,
        labels=y_val,
        tokenizer=tokenizer
    )

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    return val_loader, y_val

def load_model(model_path, device):
    model = BertWithScalarFeatures(scalar_feature_dim=16, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device, true_labels):
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scalar_feats = batch['scalar_features'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, scalar_features=scalar_feats)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    acc = accuracy_score(true_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, all_preds, average='weighted')
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_model.pt"  
    CSV_PATH = "../Data/Train/processed_annotated_train.csv"  

    val_loader, true_labels = load_data(CSV_PATH)
    model = load_model(MODEL_PATH, DEVICE)
    evaluate(model, val_loader, DEVICE, true_labels)