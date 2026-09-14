import json
import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

class RankingTripletDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        return item["query"], item["positive"], item["negative"]

def infonce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = torch.cat([pos_scores, neg_scores], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model and Tokenizer
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    
    # 2. Setup DataLoader and Optimizer
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "hard_negatives.json")
    dataset = RankingTripletDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    print("Starting Training Loop...")
    
    for epoch in range(3):  # Standard for fine-tuning
        total_loss = 0.0
        
        for queries, positives, negatives in dataloader:
            optimizer.zero_grad()
            
            # Tokenize Query + Positive pairs
            pos_inputs = tokenizer(queries, positives, padding=True, truncation=True, return_tensors="pt").to(device)
            # Tokenize Query + Negative pairs
            neg_inputs = tokenizer(queries, negatives, padding=True, truncation=True, return_tensors="pt").to(device)
            
            # Forward pass
            pos_scores = model(**pos_inputs).logits
            neg_scores = model(**neg_inputs).logits
            
            # Calculate InfoNCE Loss
            loss = infonce_loss(pos_scores, neg_scores, temperature=1.0)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1} | Average Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    main()
