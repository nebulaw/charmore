import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42069)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 10_000
eval_interval = 500
learning_rate = 1e-3
device = 'cpu'
eval_iters = 200
max_new_tokens = 500


# read galaktioni poems
with open("./data/gala.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text = text.replace("\xad", "-")


# prepare vocabulary, encode and decode functions
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
stoi = { s:i for i, s in enumerate(vocab) }
itos = { i:s for s, i in stoi.items() }
encode = lambda seq: [stoi[ch] for ch in seq]
decode = lambda key: ''.join([itos[i] for i in key])


# encode text
data = torch.tensor(encode(text), dtype=torch.long)
# split dataset
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# random batching function
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# create bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # this has shape of (B, T, C)
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T, C)
        for _ in range(max_new_tokens):
            # get preds
            logits, _ = self(idx)
            # focus only the last time step
            logits = logits[:, -1, :]
            # apply softmax for probs
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# create our bigram model
model = BigramLanguageModel(vocab_size)
# send parameters to device
m = model.to(device)

# you are a nice optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# train phase
for iter in range(max_iters):
    # every once in a while evaluate on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"on step {iter:-6}, train loss is {losses["train"]:.4f} and eval loss is {losses["val"]:.4f}")

    # sample random batch
    Xb, yb = get_batch("train")
    
    # evaluate model
    logits, loss = model(Xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from new model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))
