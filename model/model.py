import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42069)

# hyperparameters
batch_size = 64
block_size = 256 # this is context length
max_iters = 4000
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2


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


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # computing attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd), # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        # x = self.sa(x)
        # x = self.ffn(x)
        # added residual connections for better optimization
        x = x + self.sa(x) 
        x = x + self.ffn(x)
        return x


# create bigram model
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # self.ffn = FeedForward(n_embd)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (B, T, vocab_size)
        x = tok_embd + pos_embd
        # x = self.sa_heads(x) # one head of attention
        # x = self.ffn(x)
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) # decoder

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
            idx_cond = idx[:, -block_size:]
            # get preds
            logits, _ = self(idx_cond)
            # focus only the last time step
            logits = logits[:, -1, :]
            # apply softmax for probs
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# create our bigram model
model = GPTLanguageModel()
# send parameters to device
m = model.to(device)

# you are a nice optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


# train phase
for iter in range(1, max_iters):
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

    print(f"end of {iter=}, loss: {loss.item()}")


while True:
    max_new_tokens = int(input("Enter number of tokens to generate "))
    # generate from new model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))
    ui = input("Continue? [Y/n] ")
    if not ui.lower() in ["yes", "y"]:
        print("Exited")
        break
