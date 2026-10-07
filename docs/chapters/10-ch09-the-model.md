# 9. The model

::: filename
model.py
:::

``` python
"""A small GPT-style decoder in MLX (about 19 million parameters)."""
import json
from dataclasses import dataclass, asdict
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class Config:
    vocab_size: int = 459
    ctx: int = 1024          # longest sequence the model is trained on
    dim: int = 512           # width of every token vector
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.rope = nn.RoPE(c.dim // c.n_heads)      # rotary position encoding

    def __call__(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        # [B, T, C] -> [B, heads, T, C / heads]
        q, k, v = (a.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3) for a in (q, k, v))
        past = 0 if cache is None else cache[0].shape[2]
        q, k = self.rope(q, offset=past), self.rope(k, offset=past)
        if cache is not None:                        # reuse earlier keys/values
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        mask = "causal" if T > 1 else None           # a token cannot see ahead
        if cache is not None and T > 1:
            raise ValueError("with a cache, feed one token at a time")
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out(y), (k, v)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(c.dim)
        self.attn = Attention(c)
        self.norm2 = nn.RMSNorm(c.dim)
        self.fc1 = nn.Linear(c.dim, 4 * c.dim, bias=False)
        self.fc2 = nn.Linear(4 * c.dim, c.dim, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x, cache=None):
        a, cache = self.attn(self.norm1(x), cache)
        x = x + self.drop(a)
        x = x + self.drop(self.fc2(nn.gelu(self.fc1(self.norm2(x)))))
        return x, cache


class Pianist(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.dim)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = [Block(c) for _ in range(c.n_layers)]
        self.norm = nn.RMSNorm(c.dim)

    def __call__(self, tokens, cache=None):
        """tokens [B, T] -> logits [B, T, vocab], and the key/value cache."""
        x = self.drop(self.embed(tokens))
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if cache is None else cache[i])
            new_cache.append(c)
        # the output layer reuses the embedding table (weight tying)
        logits = self.norm(x) @ self.embed.weight.T
        return logits, new_cache

    def n_params(self):
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def save(self, path, **extra):
        self.save_weights(path + ".safetensors")
        json.dump({"config": asdict(self.config), **extra}, open(path + ".json", "w"))

    @staticmethod
    def load(path):
        info = json.load(open(path + ".json"))
        model = Pianist(Config(**info["config"]))
        model.load_weights(path + ".safetensors")
        model.eval()
        return model, info


if __name__ == "__main__":
    m = Pianist(Config())
    print(f"{m.n_params() / 1e6:.1f} M parameters")
    logits, _ = m(mx.zeros((2, 16), dtype=mx.int32))
    print("logits:", logits.shape)
```

The file follows the description in chapter 3 closely. Reading it from the bottom up is easiest.

**`Pianist`** is the whole model. It embeds the tokens, passes them through the blocks, normalises the result and multiplies by the transposed embedding table to get logits. That last line is the weight tying from chapter 3.

**`Block`** is one layer: normalise, attend, add; normalise, feed forward, add. The normalisation is RMSNorm, a simplified layer normalisation that most recent language models use. The activation between the two feed-forward layers is GELU.

**`Attention`** computes queries, keys and values with a single matrix (`qkv`) and splits the result three ways. It reshapes each so that the eight heads sit in their own dimension, applies the rotary position encoding to queries and keys, and calls MLX's built-in attention function with a causal mask.

## 9.1 How the cache works in this code

`Attention` is written to work in two modes.

In **training**, it receives a whole sequence and no cache. The causal mask keeps each position from seeing later ones.

In **generation**, the first call processes the request and any music so far in one pass and returns the keys and values it computed. Every call after that receives a single new token plus that cache. The new token's key and value are appended to the cache, and its query is compared against all of them. The `offset=past` argument tells the rotary encoding the true position of the new token, so the result is identical to what a full pass would give.

No mask is needed in the second mode: the only query is the newest token, and it is allowed to see everything.

## 9.2 Dropout

Dropout randomly zeroes 10% of the values during training, at two points in each layer and once after the embedding. It is a standard guard against memorisation. `model.eval()` switches it off for validation and generation, and `model.train()` switches it back on.
