# Implement the GPT2 model using MLS

Following the [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) implemented the GPT-2 model from scratch using Apple's [MLX](https://opensource.apple.com/projects/mlx/) framework that run faster on Apple Silicon.

## Setup

```bash
$ uv sync
```

## Run GPT-2 model by loading the weights from OpenAI's checkpoint

```bash
$ uv run src/mlxgpt/gpt2.py -h
usage: gpt2.py [-h] [-s {small,medium,large,xl}] -p PROMPT [-m MAX_TOKENS] [-t TEMPERATURE] [-k TOP_K]

Generate text using GPT-2 model

options:
  -h, --help            show this help message and exit
  -s {small,medium,large,xl}, --size {small,medium,large,xl}
                        Size of GPT-2 model (default: small)
  -p PROMPT, --prompt PROMPT
                        Start context/prompt for text generation
  -m MAX_TOKENS, --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: 50)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature for sampling (default: 1.0, 0.0 for greedy)
  -k TOP_K, --top-k TOP_K
                        Top-k sampling parameter (default: 25, None to disable)

$ uv run src/mlxgpt/gpt2.py -p "To be or not to be"
Loading gpt2-small (124M) model from cache...
Model loaded successfully from gpt2/GPT2-small.npz

Prompt: To be or not to be
Generating 50 tokens with temperature=1.0, top_k=25...

Generated text:
To be or not to be a part of the future. We're going to find something that really is going to bring out the real and the real."


On The Big Chill: "We made it and that will happen in our country, but it's only coming through
```
