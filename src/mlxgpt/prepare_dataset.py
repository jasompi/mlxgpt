# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script that processes the Project Gutenberg files into fewer larger files.
"""

import argparse
import glob
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


def process_files(file_paths: list[str], target_dir: str, combine: bool, max_size_mb: int = 100,
                  separator="<|endoftext|>", fallback_encoding: str = "latin1",
                  tokenizer: Optional[tiktoken.Encoding] = None, dtype: str = "int32",
                  max_files: Optional[int] = None):
    """
    Process multiple text files, optionally combining them into larger files or saving individually.

    Args:
        file_paths: List of input file paths to process
        target_dir: Directory where processed files will be saved
        combine: If True, combine files into larger chunks. If False, process individually (default: True)
        max_size_mb: Maximum size in MB for each combined file (ignored if combine=False)
        separator: Token to separate documents (default: "<|endoftext|>")
        fallback_encoding: Encoding to use if UTF-8 fails (default: "latin1")
        tokenizer: Optional tokenizer for creating .npy token files
        dtype: Data type for saved token arrays (default: "int32")
        max_files: Maximum number of output files to create (default: None, process all)

    Returns:
        Number of files created/processed
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
        # Replace single newlines with spaces, but only when preceded and followed by non-space characters
        content = re.sub(r'(?<![ \t\n])\n(?![ \t\n])', ' ', content)
        estimated_size = len(content.encode("utf-8"))

        # Tokenize if tokenizer is provided
        if tokenizer is not None:
            token_ids = tokenizer.encode(content)

        # If not combining, save each file individually
        if not combine:
            # Get the base filename
            base_name = os.path.basename(file_path)
            target_file_path = os.path.join(target_dir, base_name)

            # Save processed file
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(content)

            # Save tokenized version if tokenizer is provided
            if tokenizer is not None:
                name_without_ext = os.path.splitext(base_name)[0]
                target_token_file_path = os.path.join(target_dir, f"{name_without_ext}_tokens.npy")
                np.save(target_token_file_path, np.array(token_ids, dtype=dtype))

            file_counter += 1
            continue

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

    parser.add_argument("-i", "--input_data", type=str, nargs='+', default=["gutenberg/data/raw"],
                        help="Input data: directory path, wildcard pattern (e.g., data/*.txt), single file, or multiple files")
    parser.add_argument("-m", "--max_size_mb", type=int, default=100,
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
    parser.add_argument("-c", "--combine", action="store_true", default=None,
                        help="Combine input files into larger files. If not specified, inferred from input type (False for file lists, True for directories/wildcards)")

    args = parser.parse_args()

    if args.verify:
        print(f"Verifying tokenization in {os.path.abspath(args.output_dir)}...")
        if verify_tokenization(args.output_dir):
            print("\n✓ All token files verified successfully!")
        else:
            print("\n❌ Verification failed for some files")
    else:
        # Handle different input types: directory, wildcard pattern, single file, or multiple inputs
        input_paths = args.input_data
        all_files = []

        # Track if all inputs are direct files (for inferring combine mode)
        all_inputs_are_files = True

        for input_path in input_paths:
            if os.path.isfile(input_path):
                # Single file
                all_files.append(input_path)
            elif os.path.isdir(input_path):
                # Directory - walk through it
                all_inputs_are_files = False
                dir_files = [os.path.join(path, name)
                             for path, subdirs, files in os.walk(input_path)
                             for name in files if name.endswith((".txt", ".txt.utf8"))]
                all_files.extend(dir_files)
            else:
                # Wildcard pattern or non-existent path
                all_inputs_are_files = False
                matched_files = glob.glob(input_path, recursive=True)
                # Filter for text files if wildcard doesn't specify extension
                if not any(input_path.endswith(ext) for ext in [".txt", ".txt.utf8"]):
                    matched_files = [f for f in matched_files if f.endswith((".txt", ".txt.utf8"))]
                all_files.extend(matched_files)

        if not all_files:
            print(f"Error: No files found matching input patterns: {input_paths}")
            exit(1)

        # Infer combine mode if not explicitly set
        if args.combine is None:
            combine_mode = not all_inputs_are_files  # False for file lists, True for dirs/wildcards
        else:
            combine_mode = args.combine

        tokenizer = tiktoken.get_encoding("gpt2") if args.tokenize else None

        print(f"Processing {len(all_files)} file(s)...")
        print(f"Mode: {'Combining files' if combine_mode else 'Processing individually'}")
        if args.tokenize:
            print(f"Token dtype: {args.dtype}")
        if args.num_of_dataset:
            print(f"Maximum output files: {args.num_of_dataset}")
        file_counter = process_files(all_files, args.output_dir, combine=combine_mode,
                                     max_size_mb=args.max_size_mb, tokenizer=tokenizer,
                                     dtype=args.dtype, max_files=args.num_of_dataset)
        print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")
