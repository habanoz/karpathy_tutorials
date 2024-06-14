import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
max_iters = 5000
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text): return [stoi[ch] for ch in text]
def decode(tokens): return ''.join([itos[t] for t in tokens])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


#
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for t in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[t] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class Head(nn.Module):
    """one head self attention"""

    def __init__(self, head_size) -> None:
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        # perform weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)

        return out


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.sa_heads = MultiHead(4, n_embd//4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.sa_heads(x)  # (B,T,C)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        assert logits.shape == (B, T, vocab_size)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


xb, yb = get_batch('train')

model = BigramLM().to(device)
out, loss = model(xb, yb)
print(out.shape)


print(decode(model.generate(idx=torch.zeros((1, 8), dtype=torch.long,
      device=device), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_iters == 0:
        loss = estimate_loss()
        print(f"Step {step}; Train loss: {
              loss['train']:.4f}; Test loss {loss['test']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(model.generate(idx=torch.zeros((1, 8), dtype=torch.long,
      device=device), max_new_tokens=100)[0].tolist()))
