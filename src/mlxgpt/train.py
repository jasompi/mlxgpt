import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from mlxgpt.dataloader import create_gpt_dataloader, MLXDataLoader
from mlxgpt.model import GPTModel, GPTConfig
from mlxgpt.gpt2 import generate_text_simple, text_to_token_ids, token_ids_to_text
import tiktoken
import time
from typing import Optional

def calc_loss_batch(model: nn.Module, input_batch: mx.array, target_batch: mx.array) -> mx.array:
    logits = model(input_batch)
    loss = nn.losses.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), reduction="mean")
    return loss

def calc_loss_loader(model: nn.Module, data_loader: MLXDataLoader, num_batches: Optional[int] = None) -> float:
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(model, input_batch, target_batch)
            mx.eval(loss)  # Evaluate the loss before calling .item()
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

def train_model_simple(
    model: nn.Module,
    train_loader: MLXDataLoader,
    val_loader: MLXDataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: Optional[str],
    tokenizer: tiktoken.Encoding,
    use_compile: bool = False
) -> tuple[list[float], list[float], list[int]]:
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        logits = model(x)
        return nn.losses.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="mean")

    def step(x: mx.array, y: mx.array) -> mx.array:
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    # Compile the training step if requested
    if use_compile:
        state = [model.state, optimizer.state]
        train_step = mx.compile(step, inputs=state, outputs=state)
    else:
        train_step = step

    # Main training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            loss = train_step(input_batch, target_batch)

            tokens_seen += input_batch.size
            global_step += 1

            # Optional evaluation step
            if eval_freq > 0:
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, eval_iter)
                    # mx.eval(train_loss, val_loss)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            else:
                if global_step % 50 == 0:
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {loss:.3f}");

        # Print a sample text after each epoch if start_context is provided
        if start_context:
            generate_and_print_sample(
                model, tokenizer, start_context
            )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(
    model: nn.Module,
    train_loader: MLXDataLoader,
    val_loader: MLXDataLoader,
    eval_iter: int
) -> tuple[float, float]:
    model.eval()
    train_loss = calc_loss_loader(model, train_loader, num_batches=eval_iter)
    val_loss = calc_loss_loader(model, val_loader, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
    start_context: str
) -> None:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate GPT model")
    parser.add_argument("-s", "--save", type=str, default=None,
                        help="Name for saved model file (default: derived from load file or 'gpt-model')")
    parser.add_argument("-l", "--load", type=str, default=None,
                        help="Path to load pretrained model weights")
    parser.add_argument("-t", "--training", type=int, default=None,
                        help="Number of training epochs (if not specified with -l, model is loaded but not trained)")
    parser.add_argument("-e", "--eval", type=int, nargs='?', const=0, default=None,
                        help="Evaluation frequency during training (default: 0 = no eval during training). Use -e alone to eval after training/loading.")
    parser.add_argument("-g", "--generate", type=str, default=None,
                        help="Generate text with given start context")
    parser.add_argument("-c", "--compile", action="store_true",
                        help="Compile the training step for faster execution")
    parser.add_argument("-i", "--input", type=str, default="the-verdict.txt",
                        help="Input text file for training data (default: the-verdict.txt)")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot training and validation losses after training")

    args = parser.parse_args()

    # --- Hyperparameters ---
    batch_size = 4

    # --- Model & Tokenizer Setup ---
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab # 50257 for GPT-2 tokenizer

    GPT_CONFIG_124M = {
        "vocab_size": vocab_size,   # Vocabulary size
        "context_length": 256,      # Shortened context length (orig: 1024)
        "emb_dim": 768,             # Embedding dimension
        "n_heads": 12,              # Number of attention heads
        "n_layers": 12,             # Number of layers
        "drop_rate": 0.1,           # Dropout rate
        "qkv_bias": False           # Query-key-value bias
    }

    model = GPTModel(GPTConfig.from_dict(GPT_CONFIG_124M))

    # Load pretrained weights if specified
    if args.load:
        print(f"Loading model weights from {args.load}...")
        model.load_weights(args.load)        
        print("Model weights loaded successfully")

    mx.eval(model.parameters()) # Materialize model parameters

    num_params = sum(p.size for _, p in utils.tree_flatten(model.parameters()))
    print(f"Model initialized with {num_params:,} parameters.")

    # --- Data Loading ---
    with open(args.input, "r", encoding="utf-8") as f:
        text_data = f.read()

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_gpt_dataloader(
        train_data,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
    )

    val_loader = create_gpt_dataloader(
        val_data,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
    )

    if args.training:
        # Training mode
        optimizer = optim.AdamW(learning_rate=0.0001, weight_decay=0.01)

        # Default to 20 epochs if not specified
        num_epochs = args.training if args.training is not None else 20
        start_time = time.time()

        # Determine evaluation:
        # if -e not specified no evalutation,
        # if -e specified without value, no eval during training, only at end
        # if -e specified with value, use that value for eval during training
        # and also eval at the end
        eval_freq = 0 if args.eval is None else args.eval

        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer,
            num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=5,
            start_context=args.generate, tokenizer=tokenizer,
            use_compile=args.compile
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        # Plot losses if requested and evaluation was performed during training
        if args.plot and train_losses:
            epochs_tensor = mx.linspace(0, num_epochs, len(train_losses))
            examples_seen_tensor = mx.linspace(0, tokens_seen[-1], len(train_losses))
            plot_losses(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        # Determine save filename
        if args.save:
            base_name = args.save
        elif args.load:
            # Derive from load filename: remove extension and timestamp if present
            import os
            base_name = os.path.splitext(os.path.basename(args.load))[0]
            # Remove timestamp pattern (e.g., -20240101-123456)
            import re
            base_name = re.sub(r'-\d{8}-\d{6}$', '', base_name)
        else:
            base_name = "gpt-model"

        # Save model weights
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"{base_name}-{timestamp}.npz"
        print(f"Saving model weights to {save_path}...")
        model.save_weights(save_path)
        print(f"Model weights saved to {save_path}")

    # If -e was specified, perform eval after training
    if args.eval is not None:
        print("Running final evaluation...")
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter=10)
        print(f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

    # If -g was specified, generate text after training
    if args.generate and not args.training:
        print(f"\nGenerating text with start context: '{args.generate}'")
        generate_and_print_sample(model, tokenizer, args.generate)


if __name__ == "__main__":
    main()
