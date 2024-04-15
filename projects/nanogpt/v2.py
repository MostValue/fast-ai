import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams

batch_size = 16 # how many independent sequences
block_size = 32 # max size of context block
max_iters = 5000
eval_interval = 100 
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 64
n_head = 4 # number of attention heads
n_layer = 4 # number of non-linear layers? MLP / relu layers
dropout = 0.0 # dropout percentage

# -----------------

torch.manual_seed(1337)

with open('input.txt') as f:
    text = f.read()


# unique chars, tokeniser and encode / decode functions

chars = sorted(list(set(text)))
vocab_size = len(chars)

s_to_i = {c:i for i,c in enumerate(chars)}
i_to_s = {i:c for i,c in enumerate(chars)}

encode = lambda s: [s_to_i[c] for c in s] # output a list of integers
decode = lambda n: "".join([i_to_s[i] for i in n]) # output a string



# train and test splits

data = torch.tensor(encode(text), dtype = tensor.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (block_size,)) #drawing size of block_size from data
    xs = torch.stack([data[i:i+block_size] for i in ixs])
    ys = torch.stack([data[i+1:i+block_size+1] for i in ixs])
    x, y = x.to(device), y.to(device)
    return x, y


# loss function

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # number of iterations to evaluate over (makes loss more stable)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute attention scores
        
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T), dividing by sqrt of head size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim = -1) # (B, T, T) softmax over the rows
        wei = self.dropout(wei) 

        # perform weighted aggregation

        v = self.values(x) 
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

    

## TO IMPLEMENT / REDO

class Block(nn.Module):
    """Transformer Block"""
        """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x  # Rewatch video on this part

