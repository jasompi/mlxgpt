import mlx.core as mx
import mlx.data as mdx
import tiktoken
from typing import Iterator, Tuple, Optional, Callable

class MLXDataLoader:
    def __init__(self, dataset: mdx.Buffer, sample_transform_fn: Optional[Callable[[dict], dict]] = None, input_key: str = "input", target_key: str = "label", batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, pad: dict[str, float] = {}):
        self.batch_size = batch_size
        buffer = dataset
        if sample_transform_fn is not None:
            buffer = buffer.sample_transform(sample_transform_fn)
        buffer = buffer.shuffle() if shuffle else buffer
        buffer = buffer.batch(batch_size, pad=pad) if batch_size > 1 else buffer
        self.buffer = buffer
        self.drop_last = drop_last
        self.input_key = input_key
        self.target_key = target_key

    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        # Use the buffer for efficient streaming and batching
        for batch in self.buffer.to_stream():
            if self.drop_last and batch[self.input_key].shape[0] < self.batch_size:
                break
            yield mx.array(batch[self.input_key]), mx.array(batch[self.target_key])

    def __len__(self) -> int:
        """Return number of batches."""
        dataset_size = self.buffer.size()
        return dataset_size - 1 if self.drop_last and self.buffer[-1][self.input_key].shape[0] < self.batch_size  else dataset_size


tokenizer = tiktoken.get_encoding("gpt2")


def create_gpt_dataloader(txt: str, batch_size: int = 4, max_length: int = 256,
                          stride: int = 128, shuffle: bool = True, drop_last: bool = True) -> MLXDataLoader:
    """
    Creates an MLX data loader for a GPT-style language model.

    This function processes a text, tokenizes it, and then uses a
    sliding window to create input and target sequences. It then uses
    the MLX data library to create an efficient data stream for training.

    Args:
        txt (str): The input text to be processed.
        batch_size (int): The number of samples per batch.
        max_length (int): The maximum length of the input and target sequences.
        stride (int): The stride (step size) for the sliding window.
        shuffle (bool): Whether to shuffle the data stream.
        drop_last (bool): Whether to drop the last batch if it's smaller than the batch size.

    Returns:
        MLXDataLoader: A data loader that yields batches of (input_ids, target_ids) as MLX arrays.
    """

    # Tokenize the entire text
    token_ids = mx.array(tokenizer.encode(txt, allowed_special={"<|endoftext|>"}))
    if len(token_ids) < max_length + 1:
        raise ValueError("Number of tokenized inputs must at least be equal to max_length + 1")

    # Create a list of tuples for the sliding window
    data_list = []
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        data_list.append({"input_ids": input_chunk, "target_ids": target_chunk})

    # Create the MLX data buffer from the list of samples
    data_buffer = mdx.buffer_from_vector(data_list)

    # Return MLXDataLoader with appropriate settings
    return MLXDataLoader(
        dataset=data_buffer,
        input_key="input_ids",
        target_key="target_ids",
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def create_spam_dataset(csv_file: str, tokenizer: tiktoken.Encoding, max_length: Optional[int] = None, pad_token_id: int = tokenizer.eot_token) -> Tuple[mdx.Buffer, int]:
    """
    Create a spam dataset buffer from a CSV file.

    Args:
        csv_file: Path to the CSV file containing 'Text' and 'Label' columns
        tokenizer: Tokenizer to encode the text
        max_length: Maximum sequence length. If None, uses the longest sequence in the dataset
        pad_token_id: Token ID to use for padding

    Returns:
        Tuple of (buffer, max_length) where buffer is the MLX data buffer and max_length is the sequence length used
    """
    import pandas as pd
    data = pd.read_csv(csv_file)

    # Pre-tokenize texts
    encoded_texts = [tokenizer.encode(text) for text in data["Text"]]

    # Determine max_length
    if max_length is None:
        max_length = max(len(encoded_text) for encoded_text in encoded_texts)
    else:
        # Truncate sequences to max_length
        encoded_texts = [encoded_text[:max_length] for encoded_text in encoded_texts]

    # Get labels
    labels = data["Label"].values

    # Create buffer with data items
    data_items = [
        {"input": mx.array(encoded_text, dtype=mx.int32), "label": mx.array(label, dtype=mx.int8), "text": text.encode('utf-8')}
        for encoded_text, label, text in zip(encoded_texts, labels, data["Text"].values)
    ]

    buffer = mdx.buffer_from_vector(data_items).pad_to_size(key="input", dim=-1, size=max_length, pad_value=pad_token_id)

    return buffer, max_length

if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")

    train_buffer, max_length = create_spam_dataset(
        csv_file="validation.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    print(max_length)

    def decode_text(token_list: list[int]) -> bytes:
        """Helper function to decode tokens to text"""
        return tokenizer.decode(token_list[:token_list.index(tokenizer.eot_token)]
                                if tokenizer.eot_token in token_list else token_list).encode(encoding='utf-8')

    s = train_buffer.key_transform("input", lambda x: len(decode_text(x.tolist())), "length").to_stream()
    for i in range(5):
        sample = next(s)
        print(f"label: {sample['label']}, length: {sample['length']}, text: {sample['text'].tobytes().decode('utf-8')}")
    print(max_length)

    batch_size = 8

    # --- Combined Test for Length and Shape Validation ---
    print("\n--- Combined Test for Length and Shape Validation ---")

    # Case 1: drop_last=False
    print("\nTesting with drop_last=False...")
    dataloader_drop_false = MLXDataLoader(train_buffer, target_key="label", batch_size=batch_size, shuffle=False, drop_last=False)
    num_batches_from_len = len(dataloader_drop_false)
    dataset_size = train_buffer.size()
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}")
    print(f"len(dataloader) reports {num_batches_from_len} batches.")

    first_batch_shape = None
    all_same_shape_except_last = True
    num_batches_from_iter = 0
    for i, (input_batch, target_batch) in enumerate(dataloader_drop_false):
        num_batches_from_iter += 1
        assert isinstance(input_batch, mx.array), f"Input batch is {type(input_batch)}, expected mx.array"
        assert isinstance(target_batch, mx.array), f"Target batch is {type(target_batch)}, expected mx.array"
        current_shape = input_batch.shape
        if i == 0:
            first_batch_shape = current_shape
        elif i < num_batches_from_len - 1 and current_shape != first_batch_shape:
            all_same_shape_except_last = False
            print(f"❌ ERROR: Batch {i} has shape {current_shape}, expected {first_batch_shape}")
            break

    print(f"Iterating through dataloader yields {num_batches_from_iter} batches.")
    assert num_batches_from_len == num_batches_from_iter, "Mismatch in batch count for drop_last=False"
    if all_same_shape_except_last:
        print("✅ All batches (except possibly the last) have the same shape and batch count is verified.")

    # Case 2: drop_last=True
    print("\nTesting with drop_last=True...")
    dataloader_drop_true = MLXDataLoader(train_buffer, target_key="label", batch_size=batch_size, shuffle=False, drop_last=True)
    num_batches_from_len = len(dataloader_drop_true)
    print(f"len(dataloader) reports {num_batches_from_len} batches.")

    num_batches_from_iter = 0
    for i, (input_batch, target_batch) in enumerate(dataloader_drop_true):
        num_batches_from_iter += 1
        assert isinstance(input_batch, mx.array), f"Input batch is {type(input_batch)}, expected mx.array"
        assert isinstance(target_batch, mx.array), f"Target batch is {type(target_batch)}, expected mx.array"
        assert input_batch.shape[0] == batch_size, f"Batch {i} has size {input_batch.shape[0]}, expected {batch_size}"

    print(f"Iterating through dataloader yields {num_batches_from_iter} batches.")
    assert num_batches_from_len == num_batches_from_iter, "Mismatch in batch count for drop_last=True"
    print("✅ All batches have the correct shape and batch count is verified.")

    # --- Test sample_transform_fn ---
    print("\n--- Testing sample_transform_fn ---")

    # Define a simple transform that adds text length
    def add_text_length(sample: dict) -> dict:
        text_bytes = sample['text'].tobytes()
        sample['text_length'] = len(text_bytes)
        return sample

    dataloader_with_transform = MLXDataLoader(
        train_buffer,
        sample_transform_fn=add_text_length,
        target_key="label",
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # Verify that the transform was applied by checking the buffer directly
    print("Testing that sample_transform_fn adds 'text_length' key to samples...")
    first_batch_shape = None
    all_same_shape = True
    for i, (input_batch, target_batch) in enumerate(dataloader_with_transform):
        if i == 0:
            # Get one sample from the transformed stream to verify
            test_stream = train_buffer.sample_transform(add_text_length).to_stream()
            test_sample = next(test_stream)
            assert 'text_length' in test_sample, "Transform did not add 'text_length' key"
            print(f"✅ Transform applied successfully. Sample text length: {test_sample['text_length']}")
            first_batch_shape = input_batch.shape
        else:
            current_shape = input_batch.shape
            if i < len(dataloader_with_transform) - 1 and current_shape != first_batch_shape:
                all_same_shape = False
                print(f"❌ ERROR: Batch {i} has shape {current_shape}, expected {first_batch_shape}")
                break

    if all_same_shape:
        print("✅ All batches (except possibly the last) have the same shape.")
    print("✅ sample_transform_fn test passed.")

    # --- Test create_gpt_dataloader ---
    print("\n--- Testing create_gpt_dataloader ---")

    # Simple example text
    example_text = "Hello, my name is GPT. I am a language model. I can write and answer questions. The quick brown fox jumps over the lazy dog."

    # Create the MLX data loader
    gpt_dataloader = create_gpt_dataloader(
        txt=example_text,
        batch_size=3,
        max_length=4,
        stride=4,
        shuffle=False,
    )

    print(f"type of gpt_dataloader: {type(gpt_dataloader)}")
    print(f"len of gpt_dataloader: {len(gpt_dataloader)}")

    print("Iterating through the GPT data loader:")
    for i, (input_ids, target_ids) in enumerate(gpt_dataloader):
        print(f"\nBatch {i+1}:")
        print("Input IDs shape:", input_ids.shape)
        print("Target IDs shape:", target_ids.shape)
        assert isinstance(input_ids, mx.array), f"Input IDs is {type(input_ids)}, expected mx.array"
        assert isinstance(target_ids, mx.array), f"Target IDs is {type(target_ids)}, expected mx.array"
        assert len(input_ids.shape) == 2, f"Input IDs has {len(input_ids.shape)} dimensions, expected 2"
        assert len(target_ids.shape) == 2, f"Target IDs has {len(target_ids.shape)} dimensions, expected 2"
        assert input_ids.shape == target_ids.shape, f"Shape mismatch: input {input_ids.shape} vs target {target_ids.shape}"
        print("Input IDs:", input_ids)
        print("Target IDs:", target_ids)

        # Decode and print the text for clarity
        input_text = [tokenizer.decode(ids.tolist()) for ids in input_ids]
        target_text = [tokenizer.decode(ids.tolist()) for ids in target_ids]
        print("Decoded Input:", "\n".join([repr(line) for line in input_text]))
        print("Decoded Target:", "\n".join([repr(line) for line in target_text]))

    print("✅ create_gpt_dataloader test passed.")
