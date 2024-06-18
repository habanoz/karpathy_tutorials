from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value  projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.block_size, config.block_size)))
            .view(1, 1, config.block_size, config.block_size)
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension(n_embd)

        # calculate q uery, key and value for all heads in the batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 124M, n_heads=12, hs=64, so nh*hs=768 channels in the Transformer

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention (materializes the large (T,T) matrix for all the queries and keys)

        denominator = (1.0 / math.sqrt(k.size(-1)))
        att = (q @ k.transpose(-2, -1)) * denominator  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # (B, nh, T, hs)

        y = att @ v  # (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # tanh approximation version is specific to gpt2, approximation is not needed otherwise
        self.gelu = nn.GELU(approximate='tanh')

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # communication between tokens. aggregation pooling: reduce operation
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)  # feed forward network. map operation

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50K BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing schema
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long,
                           device=idx.device)  # shape (T)
        # position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class LiteDataLoader:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()
    
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = enc.encode(text)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        pos = self.current_pos
        buf = torch.tensor(self.tokens[pos: pos + B*T+1])

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B*T

        # if data is used up
        # revert to beginning
        if self.current_pos+B*T+1 > len(self.tokens):
            self.current_pos = 0

        return x, y
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader =  LiteDataLoader(B=2, T=1024)

    torch.set_float32_matmul_precision('high')

    #model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)
    
    
    optimizer =  torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x,y)

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T ) / (t1-t0)

        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
    