"""A small GPT-style decoder in MLX (about 19 million parameters)."""
import json
from dataclasses import dataclass, asdict
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class Config:
    """Model hyperparameters. Saved next to the weights so a checkpoint can be rebuilt."""
    vocab_size: int = 459
    ctx: int = 1024          # longest sequence the model is trained on
    dim: int = 512           # width of every token vector
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1


class Attention(nn.Module):
    """Causal multi-head self-attention with rotary positions and a key/value cache."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.rope = nn.RoPE(c.dim // c.n_heads)      # rotary position encoding

    def __call__(self, x, cache=None):
        """x [B, T, C] -> (output [B, T, C], (keys, values) to pass back in next call).

        Without a cache the whole sequence is processed at once (training).
        With a cache, feed one new token and it attends to everything before it
        (fast generation)."""
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        # [B, T, C] -> [B, heads, T, C / heads]
        q, k, v = (a.reshape(B, T, self.n_heads, -1).transpose(0, 2, 1, 3) for a in (q, k, v))
        past = 0 if cache is None else cache[0].shape[2]   # tokens already seen
        q, k = self.rope(q, offset=past), self.rope(k, offset=past)
        if cache is not None:                        # reuse earlier keys/values
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        mask = "causal" if T > 1 else None           # a token cannot see ahead
        if cache is not None and T > 1:
            raise ValueError("with a cache, feed one token at a time")
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)   # merge the heads back
        return self.out(y), (k, v)


class Block(nn.Module):
    """One transformer layer: attention then a feed-forward MLP, each pre-normed
    with RMSNorm and added back onto the residual stream."""

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
    """The full model: token embedding, a stack of Blocks, and a tied output layer."""

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
        """Total number of trainable values."""
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def save(self, path, **extra):
        """Write `path`.safetensors (weights) and `path`.json (config plus `extra`)."""
        self.save_weights(path + ".safetensors")
        json.dump({"config": asdict(self.config), **extra}, open(path + ".json", "w"))

    @staticmethod
    def load(path):
        """Rebuild a model saved with `save`, in eval mode (dropout off).
        Returns (model, info) where info is the saved json."""
        info = json.load(open(path + ".json"))
        model = Pianist(Config(**info["config"]))
        model.load_weights(path + ".safetensors")
        model.eval()
        return model, info


if __name__ == "__main__":
    # smoke test: build the default model and run a dummy batch through it
    m = Pianist(Config())
    print(f"{m.n_params() / 1e6:.1f} M parameters")
    logits, _ = m(mx.zeros((2, 16), dtype=mx.int32))
    print("logits:", logits.shape)
