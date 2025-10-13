#!/usr/bin/env python3
"""
Script to convert NPZ/NPY files to NPY format with optional dtype conversion.
"""

import argparse
import numpy as np
import os
from pathlib import Path


def convert_array(input_path: str, output_dir: str = None, target_dtype: str = None) -> tuple[str, float, int, int, str, str]:
    """
    Convert an NPZ or NPY file to NPY format with optional dtype conversion.

    Args:
        input_path: Path to the input NPZ or NPY file
        output_dir: Directory to save the NPY file (default: same as input)
        target_dtype: Target numpy dtype (e.g., 'uint16', 'float32', 'int32')

    Returns:
        Tuple of (output_path, size_ratio, input_size, output_size, original_dtype, final_dtype)
    """
    input_path_obj = Path(input_path)

    # Load the array based on file type
    if input_path_obj.suffix == ".npz":
        npz_data = np.load(input_path)
        array_keys = list(npz_data.keys())
        if not array_keys:
            raise ValueError(f"NPZ file {input_path} contains no arrays")

        array_name = array_keys[0]
        array_data = npz_data[array_name]

        if len(array_keys) > 1:
            print(f"Warning: NPZ file contains {len(array_keys)} arrays. Using '{array_name}'")
    elif input_path_obj.suffix == ".npy":
        array_data = np.load(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path_obj.suffix}")

    original_dtype = str(array_data.dtype)

    # Convert dtype if specified
    if target_dtype is not None:
        array_data = array_data.astype(target_dtype)
        final_dtype = str(array_data.dtype)
    else:
        final_dtype = original_dtype

    # Determine output path
    if output_dir is None:
        output_dir_path = input_path_obj.parent
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / (input_path_obj.stem + ".npy")

    # Save as NPY
    np.save(str(output_path), array_data)

    # Calculate file sizes and ratio
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    size_ratio = input_size / output_size if output_size > 0 else 0

    return str(output_path), size_ratio, input_size, output_size, original_dtype, final_dtype


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ/NPY files to NPY format with optional dtype conversion"
    )
    parser.add_argument(
        "input",
        nargs='+',
        help="Input NPZ/NPY file(s), wildcard pattern, or directory"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for NPY files (default: same as input)"
    )
    parser.add_argument(
        "-t", "--type",
        dest="dtype",
        default=None,
        help="Target numpy dtype (e.g., uint16, int32, float32, float64)"
    )

    args = parser.parse_args()

    # Collect input files
    input_files = []
    for input_item in args.input:
        input_path = Path(input_item)

        if input_path.is_file():
            if input_path.suffix not in [".npz", ".npy"]:
                print(f"Warning: Skipping {input_path} (not a NPZ or NPY file)")
                continue
            input_files.append(input_path)
        elif input_path.is_dir():
            dir_files = list(input_path.glob("*.npz")) + list(input_path.glob("*.npy"))
            input_files.extend(dir_files)
        else:
            print(f"Warning: {input_path} does not exist")

    if not input_files:
        print("Error: No valid NPZ/NPY files found")
        return

    print(f"Found {len(input_files)} file(s)")
    if args.dtype:
        print(f"Target dtype: {args.dtype}")
    print()

    # Process each file
    total_input_size = 0
    total_output_size = 0

    for input_file in input_files:
        try:
            output_path, ratio, input_size, output_size, orig_dtype, final_dtype = convert_array(
                str(input_file), args.output, args.dtype
            )

            total_input_size += input_size
            total_output_size += output_size

            print(f"Converted: {input_file.name}")
            print(f"  -> {Path(output_path).name}")
            print(f"  Input size: {input_size:,} bytes ({input_size / 1024 / 1024:.2f} MB)")
            print(f"  Output size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")
            if orig_dtype != final_dtype:
                print(f"  Dtype conversion: {orig_dtype} -> {final_dtype}")
            else:
                print(f"  Dtype: {orig_dtype}")
            print(f"  Size ratio: {ratio:.4f} ({ratio * 100:.2f}%)")
            size_diff = output_size - input_size
            if size_diff > 0:
                print(f"  Size change: +{size_diff:,} bytes ({(ratio - 1) * -100:.2f}% larger)")
            elif size_diff < 0:
                print(f"  Size change: {size_diff:,} bytes ({(1 - ratio) * 100:.2f}% smaller)")
            print()

        except Exception as e:
            print(f"Error processing {input_file.name}: {e}\n")

    # Print summary
    if len(input_files) > 1:
        overall_ratio = total_input_size / total_output_size if total_output_size > 0 else 0
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total input size: {total_input_size:,} bytes ({total_input_size / 1024 / 1024:.2f} MB)")
        print(f"Total output size: {total_output_size:,} bytes ({total_output_size / 1024 / 1024:.2f} MB)")
        print(f"Overall size ratio: {overall_ratio:.4f} ({overall_ratio * 100:.2f}%)")
        print(f"Total size difference: {total_output_size - total_input_size:,} bytes")


if __name__ == "__main__":
    main()