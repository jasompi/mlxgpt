import argparse
import mlx.core as mx
import mlx.nn as nn
from mlxgpt.model import GPTModel, create_gpt_model
import tiktoken
from typing import Tuple, Optional

def generate_text_simple(model: nn.Module, idx: mx.array, max_new_tokens: int, context_size: int) -> mx.array:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  
        probas = mx.softmax(logits, axis=-1)  # (batch, vocab_size)
        idx_next = mx.argmax(probas, axis=-1, keepdims=True)  # (batch, 1)
        idx = mx.concatenate([idx, idx_next], axis=1)  # (batch, n_tokens+1)
    return idx

def topk(x: mx.array, top_k: int, axis: int = -1) -> Tuple[mx.array, mx.array]:
    """
    Returns the top-k values and indices along a specified axis.
    """
    sorted_indices: mx.array = mx.argsort(-x, axis=axis)
    top_k_indices: mx.array = mx.take(sorted_indices, mx.arange(top_k), axis=axis)
    batch_indices: mx.array = mx.arange(x.shape[0])[:, None]
    broadcasted_batch_indices: mx.array = mx.broadcast_to(batch_indices, top_k_indices.shape)
    values: mx.array = x[broadcasted_batch_indices, top_k_indices]
    return values, top_k_indices

def generate(
    model: GPTModel,
    idx: mx.array,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None
) -> mx.array:
    for _ in range(max_new_tokens):
        idx_cond: mx.array = idx[:, -context_size:]
        logits: mx.array = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits: mx.array
            top_ids: mx.array
            top_logits, top_ids = topk(logits, top_k)
            min_val: mx.array = top_logits[:, -1]
            logits = mx.where(logits < min_val, mx.array(float("-inf"), logits.dtype), logits)

        idx_next: mx.array
        if temperature > 0.0:
            logits = logits / temperature
            idx_next = mx.random.categorical(logits, axis=-1, num_samples=1)
        else:
            idx_next = mx.argmax(logits, axis=-1, keepdims=True)

        if eos_id is not None and mx.any(mx.equal(idx_next, eos_id)):
            break

        idx = mx.concatenate((idx, idx_next), axis=1)
        mx.eval(idx)

    return idx

def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> mx.array:
    encoded: list[int] = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor: mx.array = mx.expand_dims(mx.array(encoded), axis=0)
    return encoded_tensor

def token_ids_to_text(token_ids: mx.array, tokenizer: tiktoken.Encoding) -> str:
    flat: mx.array = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Generate text using GPT-2 model")
    parser.add_argument("-s", "--size", type=str, choices=["small", "medium", "large", "xl"], default="small",
                        help="Size of GPT-2 model (default: small)")
    parser.add_argument("-p", "--prompt", type=str, required=True,
                        help="Start context/prompt for text generation")
    parser.add_argument("-m", "--max-tokens", type=int, default=50,
                        help="Maximum number of tokens to generate (default: 50)")
    parser.add_argument("-t", "--temperature", type=float, default=1.0,
                        help="Temperature for sampling (default: 1.0, 0.0 for greedy)")
    parser.add_argument("-k", "--top-k", type=int, default=25,
                        help="Top-k sampling parameter (default: 25, None to disable)")

    args: argparse.Namespace = parser.parse_args()

    # Create model
    model: GPTModel = create_gpt_model(args.size)

    # Initialize tokenizer
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")

    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens with temperature={args.temperature}, top_k={args.top_k}...\n")

    token_ids: mx.array = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer),
        max_new_tokens=args.max_tokens,
        context_size=model.context_length,
        top_k=args.top_k if args.top_k > 0 else None,
        temperature=args.temperature
    )

    generated_text: str = token_ids_to_text(token_ids, tokenizer)
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()