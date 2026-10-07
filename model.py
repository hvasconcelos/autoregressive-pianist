"""A small GPT-style decoder in MLX (about 19 million parameters)."""
# json: save/load the model config next to the weights
import json
# dataclass: concise config class; asdict: turn that config into a plain dict for json
from dataclasses import dataclass, asdict
# mlx.core: Apple's array library (like NumPy, but runs on the GPU and supports autodiff)
import mlx.core as mx
# mlx.nn: neural-network layers (Linear, Embedding, RMSNorm, RoPE, Dropout, ...)
import mlx.nn as nn
# tree_flatten: turns the nested dict of parameters into a flat list, used to count them
from mlx.utils import tree_flatten


# @dataclass auto-generates __init__ so Config(dim=256) etc. works with defaults below
@dataclass
class Config:
    """Model hyperparameters. Saved next to the weights so a checkpoint can be rebuilt."""
    # number of distinct tokens (Shift/Pitch/Vel/Dur + tags + specials) from tokenizer.py
    vocab_size: int = 459
    ctx: int = 1024          # longest sequence the model is trained on
    dim: int = 512           # width of every token vector
    n_layers: int = 6        # how many transformer Blocks are stacked
    n_heads: int = 8         # attention heads per layer; each head is dim / n_heads = 64 wide
    dropout: float = 0.1     # fraction of activations randomly zeroed during training


# self-attention: each token looks at earlier tokens and mixes in information from them
class Attention(nn.Module):
    """Causal multi-head self-attention with rotary positions and a key/value cache."""

    # build the layer's weights from the config
    def __init__(self, c: Config):
        # let nn.Module set up its parameter bookkeeping
        super().__init__()
        # remember the head count; needed in __call__ to split vectors into heads
        self.n_heads = c.n_heads
        # one matrix produces query, key and value at once: 512 -> 3 * 512 (split later)
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        # projects the concatenated head outputs back into the 512-wide residual stream
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        # rotary position encoding: rotates each 64-wide head vector by an angle that
        # depends on the token's position, so attention can tell how far apart tokens are
        self.rope = nn.RoPE(c.dim // c.n_heads)

    # forward pass; `cache` holds keys/values of earlier tokens during generation
    def __call__(self, x, cache=None):
        """x [B, T, C] -> (output [B, T, C], (keys, values) to pass back in next call).

        Without a cache the whole sequence is processed at once (training).
        With a cache, feed one new token and it attends to everything before it
        (fast generation)."""
        # B = batch size, T = number of tokens in this call, C = vector width (512)
        B, T, C = x.shape
        # compute q/k/v in one matmul, then cut the 1536-wide result into three 512-wide parts
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        # [B, T, C] -> [B, heads, T, C / heads]
        # split each 512-wide vector into 8 heads of 64, then move the head axis before
        # the time axis so every head attends over the sequence independently
        q, k, v = (a.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3) for a in (q, k, v))
        # tokens already seen: the cached keys have shape [B, heads, past, 64]
        past = 0 if cache is None else cache[0].shape[2]
        # apply rotary positions; offset=past so a new token gets position past, past+1, ...
        # rather than restarting at 0 (only q and k are rotated, v carries content)
        q, k = self.rope(q, offset=past), self.rope(k, offset=past)
        # reuse earlier keys/values: the new token must attend to all previous ones
        if cache is not None:
            # append the new key(s) after the cached ones along the time axis (axis 2)
            k = mx.concatenate([cache[0], k], axis=2)
            # same for values
            v = mx.concatenate([cache[1], v], axis=2)
        # a token cannot see ahead: with several tokens, apply a causal (lower-triangular)
        # mask; a single new token has nothing after it, so it needs no mask
        mask = "causal" if T > 1 else None
        # the causal mask assumes q and k start at the same position, which is false
        # once a cache is prepended, so multi-token calls with a cache are refused
        if cache is not None and T > 1:
            # tell the caller to feed generated tokens one by one
            raise ValueError("with a cache, feed one token at a time")
        # softmax(q @ k^T / sqrt(64)) @ v per head, via MLX's fused fast kernel;
        # result is [B, heads, T, 64]: each token's weighted mix of the values it looked at
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        # merge the heads back: [B, heads, T, 64] -> [B, T, heads, 64] -> [B, T, 512]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        # mix the heads with the output projection; also return the full k/v as the new cache
        return self.out(y), (k, v)


# one layer of the transformer; the model stacks n_layers of these
class Block(nn.Module):
    """One transformer layer: attention then a feed-forward MLP, each pre-normed
    with RMSNorm and added back onto the residual stream."""

    # create the sub-layers
    def __init__(self, c: Config):
        # let nn.Module set up its parameter bookkeeping
        super().__init__()
        # normalises each token vector before attention (keeps values at a stable scale)
        self.norm1 = nn.RMSNorm(c.dim)
        # the self-attention defined above
        self.attn = Attention(c)
        # normalises again before the feed-forward part
        self.norm2 = nn.RMSNorm(c.dim)
        # feed-forward expands each token vector 512 -> 2048 ...
        self.fc1 = nn.Linear(c.dim, 4 * c.dim, bias=False)
        # ... and shrinks it back 2048 -> 512
        self.fc2 = nn.Linear(4 * c.dim, c.dim, bias=False)
        # randomly zeroes activations during training to reduce overfitting (off in eval mode)
        self.drop = nn.Dropout(c.dropout)

    # forward pass through one layer
    def __call__(self, x, cache=None):
        # attend over the normalised input; get this layer's updated key/value cache back
        a, cache = self.attn(self.norm1(x), cache)
        # residual connection: add the attention result onto the unchanged input
        x = x + self.drop(a)
        # feed-forward: norm -> expand -> GELU nonlinearity -> shrink, then add back (residual)
        x = x + self.drop(self.fc2(nn.gelu(self.fc1(self.norm2(x)))))
        # return the updated token vectors and the cache for this layer
        return x, cache


# the complete network: token ids in, next-token scores out
class Pianist(nn.Module):
    """The full model: token embedding, a stack of Blocks, and a tied output layer."""

    # build all layers from the config
    def __init__(self, c: Config):
        # let nn.Module set up its parameter bookkeeping
        super().__init__()
        # keep the config so save() can write it and backends can read ctx
        self.config = c
        # lookup table: each of the 459 token ids maps to a learned 512-wide vector
        self.embed = nn.Embedding(c.vocab_size, c.dim)
        # dropout on the embeddings (training only)
        self.drop = nn.Dropout(c.dropout)
        # the stack of 6 transformer layers (a list, so MLX registers all their weights)
        self.blocks = [Block(c) for _ in range(c.n_layers)]
        # final normalisation before turning vectors into token scores
        self.norm = nn.RMSNorm(c.dim)

    # forward pass; `cache` is a list with one (keys, values) pair per layer, or None
    def __call__(self, tokens, cache=None):
        """tokens [B, T] -> logits [B, T, vocab], and the key/value cache."""
        # turn token ids into vectors: [B, T] -> [B, T, 512], then apply dropout
        x = self.drop(self.embed(tokens))
        # collect each layer's updated cache here
        new_cache = []
        # run the vectors through every layer in order
        for i, block in enumerate(self.blocks):
            # give layer i its own slice of the cache (or None when not generating)
            x, c = block(x, None if cache is None else cache[i])
            # keep that layer's keys/values for the next call
            new_cache.append(c)
        # the output layer reuses the embedding table (weight tying)
        # multiplying by the transposed table [512, 459] scores every token by how well its
        # embedding matches the final vector; saves a separate 512x459 matrix of weights
        logits = self.norm(x) @ self.embed.weight.T
        # logits [B, T, 459]: unnormalised scores for the next token at each position
        return logits, new_cache

    # count the model's size
    def n_params(self):
        """Total number of trainable values."""
        # flatten the nested parameter dict into (name, array) pairs and add up array sizes
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    # write a checkpoint to disk
    def save(self, path, **extra):
        """Write `path`.safetensors (weights) and `path`.json (config plus `extra`)."""
        # all weight arrays go into one .safetensors file
        self.save_weights(path + ".safetensors")
        # config (as a dict) plus any extra info (e.g. step, loss) go into a json file
        json.dump({"config": asdict(self.config), **extra}, open(path + ".json", "w"))

    # staticmethod: called on the class, Pianist.load(path), since no model exists yet
    @staticmethod
    def load(path):
        """Rebuild a model saved with `save`, in eval mode (dropout off).
        Returns (model, info) where info is the saved json."""
        # read the json written by save()
        info = json.load(open(path + ".json"))
        # build an empty model with the same hyperparameters it was trained with
        model = Pianist(Config(**info["config"]))
        # fill it with the trained weights
        model.load_weights(path + ".safetensors")
        # switch to inference mode: dropout stops zeroing activations
        model.eval()
        # hand back the model and the saved metadata
        return model, info


# only runs when executed directly (python model.py), not when imported
if __name__ == "__main__":
    # smoke test: build the default model and run a dummy batch through it
    # randomly initialised model with the default Config
    m = Pianist(Config())
    # report the size in millions of parameters (about 19 M)
    print(f"{m.n_params() / 1e6:.1f} M parameters")
    # a batch of 2 sequences of 16 tokens, all token id 0, just to check shapes
    logits, _ = m(mx.zeros((2, 16), dtype=mx.int32))
    # should print (2, 16, 459): one score per vocabulary entry at every position
    print("logits:", logits.shape)
