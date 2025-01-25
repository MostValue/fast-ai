import os
import math
import time
import inspect
import tiktoken
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example, iterate_examples
# -----------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)

        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # set state 
        self.current_position = 0

    def next_batch(self):
        B,T  = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position+ B*T + 1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T

        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y 
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)


        # Flash attention is a kernel that pytorch compile cannot find because it rewrites attention, and never loads the TxT matrix into memory.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        # attention

        # att = (q @ k.transpose(-2,-1)) * (1.0/ math.sqrt(k.size(-1)))
        # att = att.masked.fill(self.bias[:,:,:T, :T] == 0, )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__() # calls the parent class (nn.Module) intialiser
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # Guassian LU with tanh approx, no longer used.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x) # running x through the MLP
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,00 BPE merges = 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # The goal is to have a container that matches up with Hugging face
        # Module dict allows you to index into the module using keys
        # Embeddings is a wrapper around a tensor that allows you to access the element
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme. first and last layers are the same which saves some parameters
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights) # iterates through all submodules

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_ATTR'):
                std *= (2 * self.config.n_layer) **-0.5 # 2x because every block as a attention and MLP, so twice the number 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # approximately 1/ sqrt(incoming features) so 1/sqrt (768)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # by default bias is uniform
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
    
    def forward(self, idx, targets=None):
        # idx is of shape(B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # implict broadcasting for position embeddings across each token
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # flattens out tensor. (-1, logits.size(-1)) 
            # is of size (B * T) (flattens out to to be) (B * T, vocab size)) , targets = (B *T , )
        return logits, loss

        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(17)
torch.manual_seed(17)
torch.set_float32_matmul_precision("high")

total_batch_size = 524288

B = 16
T = 1024


assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B*T)
print(f"Total desired batch size = {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")



model = GPT(GPTConfig())
# model = GPT.from_pretrained('gpt2')
print("works lol")


num_return_sequences = 5
max_length = 30

model.eval() # set the model to evaluation model
model.to(device) # shipped to computer on GPU


# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# torch.compile sees all the pytorch code at once and keeps that in memory
model = torch.compile(model)



max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    # warming up linearly
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # if it > lr decay iters, return the min

    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and gets to 0
    return min_lr + coeff * (max_lr - min_lr)




import time
train_loader = DataLoaderLite(B=8, T=1024)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
# speeds up the optimisation using something like momentum and RMS prop
# optimises faster than SGD
max_steps = 50
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # use autocasting, from pytorch automatic mixed precision
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps

        loss_accum += loss.detach()
        # import code; code.interact(local= locals()) # you can see that the weights are not changed, but the logits are changed to b16
        loss.backward() # accumulates gradiends, so need to set grads to 0. This will add up all gradients within a batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()

    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / (dt /1000)
    print(f"step {step} | loss {loss_accum.item()} | norm: {norm:.4f} | lr: {lr:.4e} | dt: {dt:.2f} ms | tokens per sec {tokens_per_sec}") # extracts tensor into the CPU, loss normally lives on the GPU

# this overfits the single batch




print(loss)
import sys; sys.exit(0)




# tokens = enc.encode("Hello, I'm a language model,")

# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
# x = tokens.to(device)

# # B = 5, T = 8

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# while x.size(1) < max_length:
#     # forward the model to get the logits:
#     with torch.no_grad(): # reduces memory usage when you are sure you dont need the gradient
#         logits = model(x) #(B, T, vocab_size)
#         logits = logits[:, -1, :] # (B, vocab_size)
#         probs = F.softmax(logits, dim=-1) # applies the softmax function across the last dimension, the vocab size

#         # Helps keeps the model on track
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) 

#         ix = torch.multinomial(topk_probs, 1) # (B, 1) # sample 1 token
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1) # selects the actual token from token index
#         # append to sequence
#         x = torch.cat((x, xcol), dim=1)

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)

# > Hello, I'm a language model, not a program.

# So this morning I started studying for the interview in the lab. This was not
# > Hello, I'm a language model, and one of the main things that bothers me when they create languages is how easy it becomes to create something that
# > Hello, I'm a language model, and I wrote it off on the grounds that a language model would make me more fluent. But I'm not
# > Hello, I'm a language model, I really like languages. I like languages because like, they're good. And the way we talk about languages
# > Hello, I'm a language model, a language model I'm using for data modelling. All I did was test the results and then I wrote some
