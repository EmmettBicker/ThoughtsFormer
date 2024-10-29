import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from thoughtsformer import ThoughtsFormer

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, num_heads=6, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1024, d_model))

        # Generate causal mask once during initialization
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((1024, 1024)), diagonal=1).bool()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer for next token prediction
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        B, L = x.shape

        # Handle padding mask
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = None

        # Embeddings
        x = self.embedding(x)
        x = x + self.pos_embedding[:L].unsqueeze(0)

        # Apply causal mask
        causal_mask = self.causal_mask[:L, :L]

        # Transformer with causal masking
        x = self.transformer(
            x,
            src_mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        return self.output(x)

class UniversalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, num_heads=6, num_steps=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1024, d_model))
        self.time_embedding = nn.Parameter(torch.randn(num_steps, d_model))

        # Single transformer encoder layer that we'll reuse
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )

        # Output layer for next token prediction
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        B, L = x.shape

        # Handle padding mask
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = None

        # Initial embedding
        x = self.embedding(x)
        x = x + self.pos_embedding[:L].unsqueeze(0)

        # Apply causal mask
        causal_mask = torch.triu(torch.ones((L, L)), diagonal=1).bool().to(x.device)

        # Universal Transformer loop - reuse the same layer
        for step in range(self.num_steps):
            x = x + self.time_embedding[step].unsqueeze(0).unsqueeze(1)
            x = self.transformer_layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask
            )

        return self.output(x)


class TextDataset(Dataset):
    def __init__(self, split="train", max_length=128):
        # Load a small text dataset (you can change this to any dataset you prefer)
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        if not text.strip():  # Skip empty lines
            text = "empty"

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # For causal language modeling, labels are the input shifted by 1
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token prediction

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch received")

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch received")

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class LossTracker:
    def __init__(self, model_names):
        self.histories = defaultdict(list)
        self.model_names = model_names

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.histories[key].append(value)

    def plot(self, save_path=None):
      def moving_average(data, window_size=4):
          return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
      
      plt.figure(figsize=(15, 5))
      
      # Plot smoothed training losses
      plt.subplot(1, 3, 1)
      for model_name in self.model_names:
          train_key = f'{model_name}_train_loss'
          if train_key in self.histories:
              smoothed_data = moving_average(self.histories[train_key], 10)
              plt.plot(smoothed_data, label=f'{model_name.upper()} Training')
      plt.title('Training Losses (Smoothed)')
      plt.xlabel('Batch')
      plt.ylabel('Loss')
      plt.legend()
      plt.grid(True)
      
      # Plot smoothed validation losses
      plt.subplot(1, 3, 2)
      for model_name in self.model_names:
          val_key = f'{model_name}_val_loss'
          if val_key in self.histories:
              # smoothed_data = moving_average(self.histories[val_key], 4)
              plt.plot(self.histories[val_key], label=f'{model_name.upper()} Validation', marker='o')
      plt.title('Validation Losses')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.grid(True)
      
      # Plot smoothed validation accuracies
      plt.subplot(1, 3, 3)
      for model_name in self.model_names:
          val_key = f'{model_name}_val_accuracy'
          if val_key in self.histories:
              smoothed_data = self.histories[val_key]
              plt.plot(smoothed_data, label=f'{model_name.upper()} Accuracy', marker='o')
      plt.title('Validation Accuracies')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.grid(True)
      
      plt.tight_layout()
      if save_path:
          plt.savefig(save_path)
      plt.show()



def train_epoch(models, dataloader, optimizers, device, epoch, loss_tracker):
    # Set all models to training mode
    for model in models.values():
        model.train()
    
    total_losses = {name: 0.0 for name in models.keys()}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Train each model
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            
            # Reshape for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            loss.backward()
            optimizer.step()

            # Track loss
            total_losses[name] += loss.item()
            loss_tracker.update(**{f"{name}_train_loss": loss.item()})

        # Update progress bar
        progress_bar.set_postfix({
            f"{name} loss": total_losses[name] / (i + 1)
            for name in models.keys()
        })

    return {name: total_losses[name] / len(dataloader) for name in models.keys()}

def evaluate_model(model, dataloader, device, model_name):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            # Count non-ignored tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    d_model = 384
    num_heads = 6
    dropout = 0.1

    # Load datasets
    train_dataset = TextDataset("train")
    val_dataset = TextDataset("validation")
    vocab_size = train_dataset.tokenizer.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_fn)

    # Initialize models
    models = {
        "tf1": ThoughtsFormer(vocab_size, vocab_size, 1024, d_model, num_heads, 1, dropout).to(device),
        "tf2": ThoughtsFormer(vocab_size, vocab_size, 1024, d_model, num_heads, 2, dropout).to(device),
        "tf4": ThoughtsFormer(vocab_size, vocab_size, 1024, d_model, num_heads, 4, dropout).to(device),
        "ut1": UniversalTransformer(vocab_size, d_model, num_heads, 1, dropout).to(device),
        "ut2": UniversalTransformer(vocab_size, d_model, num_heads, 2, dropout).to(device),
        "ut4": UniversalTransformer(vocab_size, d_model, num_heads, 4, dropout).to(device),
    }

    # Initialize optimizers
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=learning_rate)
        for name, model in models.items()
    }

    # Training tracking
    loss_tracker = LossTracker(list(models.keys()))
    best_perplexities = {name: float('inf') for name in models.keys()}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_losses = train_epoch(
            models, train_loader, optimizers,
            device, epoch+1, loss_tracker
        )

        print("\nTraining Losses:")
        for name, loss in train_losses.items():
            print(f"{name.upper()}: {loss:.4f}")

        # Evaluate
        for name, model in models.items():
            loss, perplexity = evaluate_model(model, val_loader, device, name.upper())
            print(f"\n{name.upper()} Validation:")
            print(f"Loss: {loss:.4f}")
            print(f"Perplexity: {perplexity:.4f}")

            # Track validation metrics
            loss_tracker.update(**{
                f"{name}_val_loss": loss,
                f"{name}_val_perplexity": perplexity
            })
            
            # Save best model
            if perplexity < best_perplexities[name]:
                best_perplexities[name] = perplexity
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizers[name].state_dict(),
                    'perplexity': perplexity,
                }, f'best_{name}_model.pt')
                print(f"Saved new best {name.upper()} model with perplexity: {perplexity:.4f}")

        # Plot progress
        loss_tracker.plot(save_path=f'lm_progress_epoch_{epoch+1}.png')

    # Final report  
    print("\nTraining completed!")
    print("\nBest Validation Perplexities:")
    for name in models.keys():
        print(f"{name.upper()}: {best_perplexities[name]:.4f}")

if __name__ == "__main__":
    main()