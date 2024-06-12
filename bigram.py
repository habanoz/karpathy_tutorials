import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-2
max_iters = 3000
eval_iters = 200

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


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

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
            logits, _ = self(idx)
            # focus on last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx

xb,yb = get_batch('train')

model = BigramLM(vocab_size).to(device)
out, loss = model(xb, yb)
print(out.shape)


print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long,
      device=device), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_iters == 0:
        loss = estimate_loss()
        print(
            f"Step {step}; Train loss: {loss['train']:.4f}; Test loss {loss['test']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long,
      device=device), max_new_tokens=100)[0].tolist()))
