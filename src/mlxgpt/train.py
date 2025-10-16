import argparse
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from mlxgpt.dataloader import create_gpt_dataloader, MLXDataLoader
from mlxgpt.model import GPTModel, GPTConfig, MODEL_CONFIGS
from mlxgpt.gpt2 import generate_text_simple, text_to_token_ids, token_ids_to_text
import tiktoken
import time
from tqdm.auto import tqdm, trange
from typing import Optional, Callable
from functools import partial
import sys
import os
import shutil

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

def is_notebook() -> bool:
    """Check if code is running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def truncate_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Truncate text to max_length, replacing newlines with spaces.

    Args:
        text: The text to truncate
        max_length: Maximum length of the output text. If None, no truncation is applied.

    Returns:
        Truncated text with "..." suffix if truncated
    """
    # Replace newlines with spaces
    output_text = text.strip().replace("\n", "¬")

    # Truncate to max_length if specified
    if max_length is not None and len(output_text) > max_length:
        # Reserve space for "..."
        truncate_at = max_length - 3
        # Find the last space before truncate_at to avoid cutting words
        last_space = output_text.rfind(" ", 0, truncate_at)
        if last_space > 0:
            output_text = output_text[:last_space] + "..."
        else:
            # No space found, just hard truncate
            output_text = output_text[:truncate_at] + "..."

    return output_text
        
def create_gpt_dataloaders(
    input_file: str,
    train_ratio: float,
    batch_size: int,
    max_length: int,
    stride: int,
) -> tuple[MLXDataLoader, MLXDataLoader]:
    """
    Create training and validation dataloaders from input file.

    Args:
        input_file: Path to input file (.npy for pre-tokenized or .txt for raw text)
        train_ratio: Ratio of data to use for training (e.g., 0.9 for 90%)
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        stride: Stride for sliding window

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Check if input is a .npy file (pre-tokenized) or text file
    if input_file.endswith('.npy'):
        token_ids: mx.array = mx.load(input_file) # type: ignore

        # Split data
        split_idx = int(train_ratio * len(token_ids))
        train_data = token_ids[:split_idx]
        val_data = token_ids[split_idx:]
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            text_data = f.read()

        # Split data
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

    # Create dataloaders
    train_loader = create_gpt_dataloader(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
    )

    val_loader = create_gpt_dataloader(
        val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, val_loader

def generate_sample(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
    start_context: str
) -> str:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
    )
    model.train()
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    return decoded_text

def train_model_simple(
    model: GPTModel,
    data_files: list[str],
    create_dataloaders_fn: Callable[[str], tuple[MLXDataLoader, MLXDataLoader]],
    optimizer: optim.Optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    tokenizer: tiktoken.Encoding,
    use_compile: bool = False,
    output_dir: Optional[str] = None,
    save_ckpt_freq: int = 100000,
    model_size_m: float = 0.0,
    start_context: Optional[str] = None,
    print_sample_iter: int = 1000
) -> tuple[list[float], list[float], list[int]]:
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    total_iter = num_epochs * len(data_files)

    # Create output directory if checkpoint saving is enabled
    if output_dir and save_ckpt_freq > 0:
        os.makedirs(output_dir, exist_ok=True)

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

    stats_bar = tqdm(total=0, desc=f"(Step {global_step + 1:09d}): Train loss ---, Val loss ---",
                                         position=0, leave=False)
    if is_notebook():
        output_fn = print  # Use standard print function in notebooks
    else:
        # Get terminal width for dynamic display sizing
        terminal_width = shutil.get_terminal_size().columns
        # Setup text output progress bar (always create)
        text_output = tqdm(desc=start_context or "", position=4, bar_format='{desc}', ncols=terminal_width, leave=True)
        # Redirect stdout to text_output only if not in a notebook
        output_fn = lambda text: text_output.set_description_str(truncate_text(text, max_length=terminal_width))
        
    try:
        # Main training loop
        for epoch in trange(num_epochs, desc="Epoch", position=1, leave=False):
            model.train()  # Set model to training mode

            # Iterate over data files
            for file_idx, data_file in enumerate(tqdm(data_files, desc="Data files", position=2, leave=False)) if len(data_files) > 1 else enumerate(data_files):
                # Create dataloaders for this data file
                train_loader, val_loader = create_dataloaders_fn(data_file)

                stats_bar.total = global_step + len(train_loader) * total_iter
                stats_bar.refresh()
                total_iter -= 1
                for input_batch, target_batch in tqdm(train_loader, desc=f"Train batch", position=3, leave=False):
                    loss = train_step(input_batch, target_batch)

                    tokens_seen += input_batch.size
                    global_step += 1
                    stats_bar.update(1)
    
                    # Save checkpoint periodically
                    if output_dir and save_ckpt_freq > 0 and global_step > 0 and global_step % save_ckpt_freq == 0:
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        checkpoint_name = f"gpt_{model_size_m:.0f}M_{global_step:06d}_{timestamp}.npz"
                        checkpoint_path = os.path.join(output_dir, checkpoint_name)
                        output_fn(f"Saving checkpoint to {checkpoint_path}...")
                        model.save_weights(checkpoint_path)
                        output_fn(f"Checkpoint saved at step {global_step}")

                    # Optional evaluation step
                    if eval_freq > 0:
                        if global_step > 0 and global_step % eval_freq == 0:
                            train_loss, val_loss = evaluate_model(
                                model, train_loader, val_loader, eval_iter)
                            # mx.eval(train_loss, val_loss)
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                            track_tokens_seen.append(tokens_seen)
                            stats_bar.set_description_str(f"(Step {global_step:09d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    if start_context and print_sample_iter > 0 and global_step > 0 and global_step % print_sample_iter == 0:
                        sample_text = generate_sample(model, tokenizer, start_context)
                        output_fn(sample_text)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

        # Save emergency checkpoint
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            interrupt_checkpoint_name = f"gpt_{model_size_m:.0f}M_interrupted_{global_step:06d}_{timestamp}.npz"
            interrupt_checkpoint_path = os.path.join(output_dir, interrupt_checkpoint_name)
            print(f"Saving interrupted checkpoint to {interrupt_checkpoint_path}...")
            os.makedirs(output_dir, exist_ok=True)
            model.save_weights(interrupt_checkpoint_path)
            print(f"Checkpoint saved at step {global_step}")

        # Print unprocessed files
        remaining_files = data_files[file_idx + 1:]
        if remaining_files:
            print(f"\nData files not yet processed ({len(remaining_files)}):")
            for f in remaining_files:
                print(f"  - {f}")
        else:
            print("\nAll data files were processed in this epoch.")

        print(f"Completed {epoch + 1} epoch(s) out of {num_epochs}")
        print(f"Total steps: {global_step + 1}")
        print(f"Total tokens seen: {tokens_seen:,}")

    return train_losses, val_losses, track_tokens_seen

def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate GPT model")
    parser.add_argument("-o", "--output_dir", type=str, default="model_checkpoints",
                        help="Directory to save model checkpoints (default: model_checkpoints)")
    parser.add_argument("--save_ckpt_freq", type=int, default=10000,
                        help="Save checkpoint every N steps (default: 10000)")
    parser.add_argument("-l", "--load", type=str, default=None,
                        help="Path to load pretrained model weights")
    parser.add_argument("-t", "--training", type=int, default=None,
                        help="Number of training epochs (if not specified with -l, model is loaded but not trained)")
    parser.add_argument("-e", "--eval_freq", type=int, default=100,
                        help="Evaluation frequency during training in steps (default: 100). Set to 0 to disable evaluation during training.")
    parser.add_argument("-g", "--generate", type=str, default=None,
                        help="Generate text with given start context")
    parser.add_argument("--print_sample_iter", type=int, default=10000,
                        help="Generate and print sample text every N iterations (default: 10000)")
    parser.add_argument("-c", "--compile", action="store_true",
                        help="Compile the training step for faster execution")
    parser.add_argument("-i", "--input", type=str, nargs='+', default=["the-verdict.txt"],
                        help="Input data file(s) for training. Accepts single file, multiple files, or wildcards (e.g., '*.npy' or 'data/*.txt')")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot training and validation losses after training")
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-4,
                        help="Learning rate for the optimizer (default: 5e-4)")
    parser.add_argument("-b", "--batch_size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument("-s", "--size", type=str, default=None,
                        help="Model size: 'small' (124M), 'medium' (355M), 'large' (774M), 'xl' (1558M). If not specified, uses debug config.")

    args = parser.parse_args()

    # Resolve wildcards and collect all input files
    data_files = []
    for pattern in args.input:
        # Expand wildcards
        matched_files = glob.glob(pattern)
        if matched_files:
            data_files.extend(matched_files)
        else:
            # If no match, treat as literal filename (could be error or single file)
            data_files.append(pattern)

    # Remove duplicates and sort
    data_files = sorted(set(data_files))

    if not data_files:
        print("Error: No input files found.")
        return

    print(f"Found {len(data_files)} data file(s):")

    # --- Hyperparameters ---
    batch_size = args.batch_size

    # --- Model & Tokenizer Setup ---
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab # 50257 for GPT-2 tokenizer

    # Select model configuration based on size parameter
    if args.size and args.size in MODEL_CONFIGS:
        # Use predefined model config and override drop_rate and qkv_bias
        config = MODEL_CONFIGS[args.size]
        config.drop_rate = 0.0
        config.qkv_bias = False
        print(f"Using '{args.size}' model configuration")
    else:
        # Use debug configuration
        if args.size:
            print(f"Warning: Size '{args.size}' not found in MODEL_CONFIGS. Using debug config.")
        GPT_CONFIG_DEBUG = {
            "vocab_size": vocab_size,
            "context_length": 128,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.0,
            "qkv_bias": False
        }
        config = GPTConfig.from_dict(GPT_CONFIG_DEBUG)
        print("Using debug model configuration")

    model = GPTModel(config)

    # Load pretrained weights if specified
    if args.load:
        print(f"Loading model weights from {args.load}...")
        model.load_weights(args.load)        
        print("Model weights loaded successfully")

    mx.eval(model.parameters()) # Materialize model parameters

    num_params = sum(p.size for _, p in utils.tree_flatten(model.parameters()))
    model_size_m = num_params / 1_000_000  # Convert to millions
    print(f"Model initialized with {num_params:,} parameters ({model_size_m:.1f}M).")

    # --- Data Loading ---
    # Determine train ratio based on file type
    train_ratio = 0.95

    if args.training:
        # Training mode
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

        # Default to 20 epochs if not specified
        num_epochs = args.training if args.training is not None else 20
        start_time = time.time()

        # Use the eval_freq from command line arguments
        eval_freq = args.eval_freq

        # Create partial function for dataloaders
        create_dataloaders_fn = partial(
            create_gpt_dataloaders,
            train_ratio=train_ratio,
            batch_size=batch_size,
            max_length=model.context_length,
            stride=model.context_length
        )

        train_losses, val_losses, tokens_seen = train_model_simple(
            model=model,
            data_files=data_files,  # Pass resolved list of files
            create_dataloaders_fn=create_dataloaders_fn,
            optimizer=optimizer,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
            eval_iter=5,
            tokenizer=tokenizer,
            use_compile=args.compile,
            output_dir=args.output_dir,
            save_ckpt_freq=args.save_ckpt_freq,
            model_size_m=model_size_m,
            start_context=args.generate,
            print_sample_iter=args.print_sample_iter
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        # Plot losses if requested and evaluation was performed during training
        if args.plot and train_losses:
            epochs_tensor = mx.linspace(0, num_epochs, len(train_losses))
            examples_seen_tensor = mx.linspace(0, tokens_seen[-1], len(train_losses))
            plot_losses(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        # Save final model weights
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        final_checkpoint_name = f"gpt_{model_size_m:.0f}M_final_{timestamp}.npz"
        final_checkpoint_path = os.path.join(args.output_dir, final_checkpoint_name)
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving final model weights to {final_checkpoint_path}...")
        model.save_weights(final_checkpoint_path)
        print(f"Final model weights saved to {final_checkpoint_path}")

    # If -g was specified, generate text after training
    if args.generate and not args.training:
        print(f"\nGenerating text with start context: '{args.generate}'")
        generated_text = generate_sample(model, tokenizer, args.generate)
        print(generated_text)


if __name__ == "__main__":
    main()
