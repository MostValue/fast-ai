import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
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

dir = Path("/home/kyan/git/fast-ai/projects/nanogpt/")
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(dir/"input.txt", 'r', encoding='utf-8') as f:
    text = f.read()



# unique chars, tokeniser and encode / decode functions

chars = sorted(list(set(text)))
vocab_size = len(chars)

s_to_i = {c:i for i,c in enumerate(chars)}
i_to_s = {i:c for i,c in enumerate(chars)}

encode = lambda s: [s_to_i[c] for c in s] # output a list of integers
decode = lambda n: "".join([i_to_s[i] for i in n]) # output a string



# train and test splits

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (block_size,)) #drawing size of block_size from data
    x = torch.stack([data[i:i+block_size] for i in ixs])
    y = torch.stack([data[i+1:i+block_size+1] for i in ixs])
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

        v = self.value(x) 
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # applies a linear transformation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out




class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity. This is a simple MLP, which represents the computation parts """

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
    """This is a transformer Block"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head  
        self.sa = MultiHeadAttention(n_head, head_size) # MHA layer
        self.ffwd = FeedFoward(n_embd) # followed by computation MLPS
        self.ln1 = nn.LayerNorm(n_embd) # Layernorm
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # allows for the adjustment of weights easier 
        x = x + self.ffwd(self.ln2(x))
        return x  



# Bigram Model

class GPT(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) #final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)


    # better init weights function
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): # instead of random it will be normally distributed
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape # Batch size, by T size of context

        tok_emb = self.token_embedding_table(idx) # BTC now includes size C of embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T,C
        x = tok_emb + pos_emb # BTC
        x = self.blocks(x)  # running through all transformer blocks
        x = self.ln_f(x) # BTC final normalisation layer
        logits = self.lm_head(x) # B, T, Vocab Size (For each time t, probability stored in vocab size)

        if targets is None: loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idxs is (B,T) array of indices for current context
        
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens (otherwise we index out of embedding array)
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # select only last time step

            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax

            probs = F.softmax(logits, dim = -1) # BTC

            # sample from distribution

            idx_next = torch.multinomial(probs, num_samples= 1) #B,1

            idx = torch.cat((idx, idx_next), dim = 1) # B, T+1

        return idx

model = GPT()
m = model.to(device)

# print number of parameters in model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))