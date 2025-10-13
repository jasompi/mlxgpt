# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script that processes the Project Gutenberg files into fewer larger files.
"""

import argparse
import numpy as np
import os
import re
import tiktoken
from tqdm import tqdm
from gutenberg.src.cleanup import strip_headers
from typing import Optional


def is_english(text: str, threshold: float = 0.9) -> bool:
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold


def combine_files(file_paths: list[str], target_dir: str, max_size_mb: int = 500, separator="<|endoftext|>",
                  fallback_encoding: str = "latin1", tokenizer: Optional[tiktoken.Encoding] = None,
                  dtype: str = "int32", max_files: Optional[int] = None):
    """
    Combine multiple text files into larger files, optionally tokenizing them.

    Args:
        file_paths: List of input file paths to combine
        target_dir: Directory where combined files will be saved
        max_size_mb: Maximum size in MB for each combined file
        separator: Token to separate documents (default: "<|endoftext|>")
        fallback_encoding: Encoding to use if UTF-8 fails (default: "latin1")
        tokenizer: Optional tokenizer for creating .npy token files
        dtype: Data type for saved token arrays (default: "int32")
        max_files: Maximum number of output files to create (default: None, process all)

    Returns:
        Number of combined files created
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Accumulators for text content
    current_content: list[str] = []  # List of document contents
    current_size: int = 0  # Accumulated size in bytes

    # Accumulators for tokenized content
    current_token_ids: list[int] = []  # Accumulated token IDs
    current_token_size: int = 0  # Accumulated token count
    token_ids: list[int] = []  # Token IDs for current document

    file_counter: int = 1  # Counter for output files

    for file_path in tqdm(file_paths):
        # Read file with UTF-8 encoding, fall back to alternative if needed
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Attempt to read the file with a fallback encoding
            tqdm.write(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        # Skip non-English content
        if not is_english(content):
            tqdm.write(f"Skipping {file_path} as it does not contain primarily English text.")
            continue

        # Remove Project Gutenberg headers/footers
        content = strip_headers(content)

        # Normalize whitespace: replace multiple blank lines with a single blank line
        content = re.sub(r'\n\s*\n', '\n\n', content)
        estimated_size = len(content.encode("utf-8"))

        # Tokenize if tokenizer is provided
        if tokenizer is not None:
            token_ids = tokenizer.encode(content)

        # Check if adding this document exceeds the size limit
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            # Check if we've reached the maximum number of files before saving
            if max_files is not None and file_counter > max_files:
                break

            # Save accumulated content to file
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content) + separator)

            # Start new accumulation with current document
            current_content = [content]
            current_size = estimated_size

            # Save tokenized version if tokenizer is provided
            if tokenizer is not None:
                target_token_file_path = os.path.join(target_dir, f"combined_{file_counter}_tokens.npy")
                np.save(target_token_file_path, np.array(current_token_ids, dtype=dtype))
                # Start new token accumulation with current document + EOT token
                current_token_ids = token_ids + [tokenizer.eot_token]

            file_counter += 1
        else:
            # Add document to current accumulation
            current_content.append(content)
            current_size += estimated_size

            # Add tokens to current accumulation if tokenizer is provided
            if tokenizer is not None:
                current_token_ids.extend(token_ids)
                current_token_ids.append(tokenizer.eot_token)  # Add end-of-text token
                

    # Save any remaining content (if we haven't reached the limit)
    num_files_saved = file_counter - 1
    if current_content and (max_files is None or file_counter <= max_files):
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content) + separator)

        # Save remaining tokens if tokenizer is provided
        if tokenizer is not None:
            target_token_file_path = os.path.join(target_dir, f"combined_{file_counter}_tokens.npy")
            np.save(target_token_file_path, np.array(current_token_ids, dtype=dtype))

        num_files_saved = file_counter

    return num_files_saved


def verify_tokenization(output_dir: str) -> bool:
    """Verify that .npy token files match the tokenization of .txt files."""
    tokenizer = tiktoken.get_encoding("gpt2")
    txt_files = sorted([f for f in os.listdir(output_dir) if f.startswith("combined_") and f.endswith(".txt")])

    if not txt_files:
        print(f"No combined_*.txt files found in {output_dir}")
        return False

    all_verified = True
    for txt_file in txt_files:
        # Extract file number
        match = re.match(r"combined_(\d+)\.txt", txt_file)
        if not match:
            continue
        file_num = match.group(1)

        txt_path = os.path.join(output_dir, txt_file)
        npy_path = os.path.join(output_dir, f"combined_{file_num}_tokens.npy")

        if not os.path.exists(npy_path):
            print(f"❌ {txt_file}: Missing corresponding tokens file")
            all_verified = False
            continue

        # Read and tokenize the text file
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the text
        expected_tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Load saved tokens
        saved_tokens = np.load(npy_path).tolist()

        # Compare
        if expected_tokens == saved_tokens:
            print(f"✓ {txt_file}: Tokens match ({len(expected_tokens)} tokens)")
        else:
            print(f"❌ {txt_file}: Token mismatch!")
            print(f"   Expected {len(expected_tokens)} tokens, got {len(saved_tokens)} tokens")
            all_verified = False

    return all_verified


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")

    parser.add_argument("-d", "--data_dir", type=str, default="gutenberg/data/raw",
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("-m", "--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("-o", "--output_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved")
    parser.add_argument("-t", "--tokenize", action="store_true",
                        help="Whether to tokenize the data after preprocessing")
    parser.add_argument("--type", dest="dtype", type=str, default="int32",
                        help="Data type for saved token arrays (e.g., int32, uint16, int64). Default: int32")
    parser.add_argument("-n", "--num_of_dataset", type=int, default=None,
                        help="Maximum number of output dataset files to create (default: None, process all)")
    parser.add_argument("-v", "--verify", action="store_true",
                        help="Verify that token files match the text files in output_dir")

    args = parser.parse_args()

    if args.verify:
        print(f"Verifying tokenization in {os.path.abspath(args.output_dir)}...")
        if verify_tokenization(args.output_dir):
            print("\n✓ All token files verified successfully!")
        else:
            print("\n❌ Verification failed for some files")
    else:
        all_files = [os.path.join(path, name)
                     for path, subdirs, files in os.walk(args.data_dir)
                     for name in files if name.endswith((".txt", ".txt.utf8"))]

        tokenizer = tiktoken.get_encoding("gpt2") if args.tokenize else None

        print(f"Processing {len(all_files)} file(s)...")
        if args.tokenize:
            print(f"Token dtype: {args.dtype}")
        if args.num_of_dataset:
            print(f"Maximum output files: {args.num_of_dataset}")
        file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb,
                                     tokenizer=tokenizer, dtype=args.dtype, max_files=args.num_of_dataset)
        print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")
