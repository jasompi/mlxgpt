import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import numpy as np

@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for backward compatibility"""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "drop_rate": self.drop_rate,
            "qkv_bias": self.qkv_bias,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GPTConfig':
        """Create config from dictionary"""
        return cls(
            vocab_size=config_dict["vocab_size"],
            context_length=config_dict["context_length"],
            emb_dim=config_dict["emb_dim"],
            n_heads=config_dict["n_heads"],
            n_layers=config_dict["n_layers"],
            drop_rate=config_dict["drop_rate"],
            qkv_bias=config_dict["qkv_bias"],
        )

# Base configuration for GPT-2
GPT_CONFIG_BASE: Dict[str, Any] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

# Model size configurations
MODEL_CONFIGS: Dict[str, GPTConfig] = {
    "small": GPTConfig(vocab_size=50257, context_length=1024, emb_dim=768, n_heads=12, n_layers=12, drop_rate=0.1, qkv_bias=True),
    "medium": GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1024, n_heads=16, n_layers=24, drop_rate=0.1, qkv_bias=True),
    "large": GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1280, n_heads=20, n_layers=36, drop_rate=0.1, qkv_bias=True),
    "xl": GPTConfig(vocab_size=50257, context_length=1024, emb_dim=1600, n_heads=25, n_layers=48, drop_rate=0.1, qkv_bias=True),
}

MODEL_SIZES: Dict[str, str] = {
    "small": "124M",
    "medium": "355M",
    "large": "774M",
    "xl": "1558M",
}

def _assign(left: mx.array, right: np.ndarray) -> mx.array:
    """Assign numpy array to MLX array with shape checking"""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return mx.array(right)

def _load_weights_into_gpt(gpt: 'GPTModel', params: Dict[str, Any]) -> None:
    """Load pretrained weights into GPT model"""
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks.layers[b].att.W_query.weight = _assign(
            gpt.trf_blocks.layers[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks.layers[b].att.W_key.weight = _assign(
            gpt.trf_blocks.layers[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks.layers[b].att.W_value.weight = _assign(
            gpt.trf_blocks.layers[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks.layers[b].att.W_query.bias = _assign(
            gpt.trf_blocks.layers[b].att.W_query.bias, q_b)
        gpt.trf_blocks.layers[b].att.W_key.bias = _assign(
            gpt.trf_blocks.layers[b].att.W_key.bias, k_b)
        gpt.trf_blocks.layers[b].att.W_value.bias = _assign(
            gpt.trf_blocks.layers[b].att.W_value.bias, v_b)

        gpt.trf_blocks.layers[b].att.out_proj.weight = _assign(
            gpt.trf_blocks.layers[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks.layers[b].att.out_proj.bias = _assign(
            gpt.trf_blocks.layers[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks.layers[b].ff.layers.layers[0].weight = _assign(
            gpt.trf_blocks.layers[b].ff.layers.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks.layers[b].ff.layers.layers[0].bias = _assign(
            gpt.trf_blocks.layers[b].ff.layers.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks.layers[b].ff.layers.layers[2].weight = _assign(
            gpt.trf_blocks.layers[b].ff.layers.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks.layers[b].ff.layers.layers[2].bias = _assign(
            gpt.trf_blocks.layers[b].ff.layers.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks.layers[b].norm1.scale = _assign(
            gpt.trf_blocks.layers[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks.layers[b].norm1.shift = _assign(
            gpt.trf_blocks.layers[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks.layers[b].norm2.scale = _assign(
            gpt.trf_blocks.layers[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks.layers[b].norm2.shift = _assign(
            gpt.trf_blocks.layers[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])

def create_gpt_model(size: str) -> 'GPTModel':
    """
    Factory method to create and load a pretrained GPT-2 model.

    Args:
        size: Model size ("small", "medium", "large", "xl")

    Returns:
        GPTModel with pretrained weights loaded
    """
    config: GPTConfig = MODEL_CONFIGS[size]
    model_size: str = MODEL_SIZES[size]

    # Check if cached weights exist
    weights_path = Path(f"gpt2/GPT2-{size}.npz")

    gpt: GPTModel = GPTModel(config)

    if weights_path.exists():
        print(f"Loading gpt2-{size} ({model_size}) model from cache...")
        gpt.load_weights(str(weights_path))
        print(f"Model loaded successfully from {weights_path}")
    else:
        from mlxgpt.gpt_download import download_and_load_gpt2
        print(f"Downloading gpt2-{size} ({model_size}) model...")
        _, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        _load_weights_into_gpt(gpt, params)

        # Save weights for future use
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        gpt.save_weights(str(weights_path))
        print(f"Model weights saved to {weights_path}")
        print(f"Model loaded successfully.")

    return gpt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self._mask = mx.tril(mx.ones((context_length, context_length)), k=0)

    def __call__(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(0, 2, 1, 3)
        queries = queries.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(0, 1, 3, 2)  # Dot product for each head

        # Use the mask to fill attention scores
        attn_scores = mx.where(self._mask[:num_tokens, :num_tokens] > 0, attn_scores, -mx.inf) 

        attn_weights = mx.softmax(attn_scores / keys.shape[-1]**0.5, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(0, 2, 1, 3)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return 0.5 * x * (1 + mx.tanh(
            mx.sqrt(mx.array(2.0 / mx.pi)) * 
            (x + 0.044715 * mx.power(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def __call__(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def __call__(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = mx.ones(emb_dim)
        self.shift = mx.zeros(emb_dim)

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True, ddof=0)
        norm_x = (x - mean) / mx.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

    def parameters(self):
        """Make scale and shift trainable parameters"""
        return {"scale": self.scale, "shift": self.shift}

class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.context_length = cfg.context_length
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg.to_dict()) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False
        )

    def __call__(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(mx.arange(seq_len))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits