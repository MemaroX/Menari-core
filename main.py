
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from collections import defaultdict
from tqdm import tqdm

# --- 1. Import Core Components ---
from small_language_model.small_language_model import SmallLanguageModel
from text_generation.text_generation import greedy_decode
from training.loss_functions import CrossEntropyLoss
from training.optimizers import Adam

# --- 2. Helper Classes for Training ---

class BasicTokenizer:
    """A simple tokenizer to handle vocabulary and encoding."""
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        """Builds a vocabulary from a given text."""
        tokens = sorted(list(set(text.split())))
        self.word_to_id = {word: i for i, word in enumerate(tokens)}
        self.id_to_word = {i: word for i, word in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def encode(self, text):
        """Converts text to a list of token IDs."""
        return [self.word_to_id[word] for word in text.split()]

    def decode(self, token_ids):
        """Converts a list of token IDs back to text."""
        return " ".join([self.id_to_word.get(tid, "<unk>") for tid in token_ids])

class DataLoader:
    """A simple data loader to yield batches of sequences."""
    def __init__(self, data, seq_len, batch_size=1):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = (len(data) - 1) // seq_len

    def __iter__(self):
        for i in range(self.num_batches):
            start_idx = i * self.seq_len
            end_idx = start_idx + self.seq_len
            inputs = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
            targets = torch.tensor(self.data[start_idx + 1 : end_idx + 1], dtype=torch.long)
            yield inputs.unsqueeze(0), targets.unsqueeze(0) # Add batch dimension


# --- 3. Main Execution ---

if __name__ == "__main__":
    print("--- Menari-core Integration: Initializing System ---")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    VOCAB_SIZE = 0 # Will be set by tokenizer
    D_MODEL = 64
    NUM_HEADS = 4
    D_FF = 128
    NUM_LAYERS = 2
    MAX_SEQ_LEN = 128
    DROPOUT_RATE = 0.1 # New hyperparameter for regularization
    EPOCHS = 10
    LEARNING_RATE = 0.001 # Adjusted for Adam

    # --- Training Data ---
    with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
        training_text = f.read()
    
    # --- Tokenization ---
    print("\n--- Step 1: Tokenization ---")
    tokenizer = BasicTokenizer()
    tokenizer.build_vocab(training_text)
    VOCAB_SIZE = tokenizer.vocab_size
    encoded_text = tokenizer.encode(training_text)
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    # print(f"Encoded Text: {encoded_text}") # Too long to print

    # --- Model Initialization ---
    print("\n--- Step 2: Initializing the Small Language Model ---")
    model = SmallLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    print("Model created successfully.")

    # --- Training Setup ---
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=LEARNING_RATE)

    # Prepare data for training (simple next-word prediction)
    data_loader = DataLoader(encoded_text, MAX_SEQ_LEN)

    # --- Training Loop ---
    print("\n--- Step 3: Commencing Training Loop ---")
    for epoch in range(EPOCHS):
        total_loss = 0

        # wrap data_loader with tqdm
        progress_bar = tqdm(
            enumerate(data_loader), 
            total=data_loader.num_batches, 
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for batch_idx, (inputs, targets) in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 1. Forward pass
            look_ahead_mask = torch.triu(torch.ones(inputs.size(1), inputs.size(1)), diagonal=1).bool().to(device)
            predictions = model(inputs, look_ahead_mask)

            # 2. Calculate loss
            loss = loss_fn.calculate_loss(predictions, targets)
            total_loss += loss.item()
            
            # 3. Backward pass
            optimizer.zero_grad()
            loss.backward()

            # 4. Update weights
            optimizer.step()

            # update the description dynamically
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / data_loader.num_batches
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    print("\n--- Training Complete ---")

    # --- Text Generation ---
    print("\n--- Step 4: Generating Text ---")
    start_token_id = tokenizer.word_to_id['the']
    generated_output = greedy_decode(model, start_token_id, max_length=10, tokenizer=tokenizer, device=device)
    
    

    print(f"\nInput prompt: 'the'")
    print(f"Generated text: {generated_output}")
    print("\n--- Menari-core Integration Complete ---")
