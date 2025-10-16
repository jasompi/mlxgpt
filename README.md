# Implement the GPT2 model using MLX framework

Following the [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) implemented the GPT-2 model from scratch using Apple's [MLX](https://opensource.apple.com/projects/mlx/) framework that run faster on Apple Silicon.

## Setup

```bash
$ uv sync
```

## Pretraining GPT-2 Model Using a Text Corpus

The train.py can be used to train a GPT2 architecture model using a text corpus from scratch.
The text corpus can be a set of txt files or pre-tokenized text files in npy format. 


```bash
$ uv run src/mlxgpt/train.py -h
usage: train.py [-h] [-o OUTPUT_DIR] [--save_ckpt_freq SAVE_CKPT_FREQ] [-l LOAD] [-t TRAINING] [-e EVAL_FREQ] [-g GENERATE] [--print_sample_iter PRINT_SAMPLE_ITER]
                [-c] [-i INPUT [INPUT ...]] [-p] [--lr LR] [-b BATCH_SIZE] [-s SIZE]

Train or evaluate GPT model

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to save model checkpoints (default: model_checkpoints)
  --save_ckpt_freq SAVE_CKPT_FREQ
                        Save checkpoint every N steps (default: 10000)
  -l LOAD, --load LOAD  Path to load pretrained model weights
  -t TRAINING, --training TRAINING
                        Number of training epochs (if not specified with -l, model is loaded but not trained)
  -e EVAL_FREQ, --eval_freq EVAL_FREQ
                        Evaluation frequency during training in steps (default: 100). Set to 0 to disable evaluation during training.
  -g GENERATE, --generate GENERATE
                        Generate text with given start context
  --print_sample_iter PRINT_SAMPLE_ITER
                        Generate and print sample text every N iterations (default: 10000)
  -c, --compile         Compile the training step for faster execution
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input data file(s) for training. Accepts single file, multiple files, or wildcards (e.g., '*.npy' or 'data/*.txt')
  -p, --plot            Plot training and validation losses after training
  --lr LR, --learning_rate LR
                        Learning rate for the optimizer (default: 5e-4)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training (default: 4)
  -s SIZE, --size SIZE  Model size: 'small' (124M), 'medium' (355M), 'large' (774M), 'xl' (1558M). If not specified, uses debug config.
```

### Pretraining GPT using Project Gutenberg Dataset

You can using the free books provided by Project Gutenberg as text corpus. To download the books clone [pgcorpus/gutenberg](https://github.com/pgcorpus/gutenberg) and folloing instructions in [README](https://github.com/pgcorpus/gutenberg/blob/master/README.md) to download and processing the data. Then use the prepare_dataset.py to combine the books and pre-tokenize the text.

```bash
$ uv run src/mlxgpt/prepare_dataset.py -h
usage: prepare_dataset.py [-h] [-i INPUT_DATA [INPUT_DATA ...]] [-m MAX_SIZE_MB] [-o OUTPUT_DIR] [-t] [--type DTYPE] [-n NUM_OF_DATASET] [-v] [-c]

Preprocess and combine text files for pretraining

options:
  -h, --help            show this help message and exit
  -i INPUT_DATA [INPUT_DATA ...], --input_data INPUT_DATA [INPUT_DATA ...]
                        Input data: directory path, wildcard pattern (e.g., data/*.txt), single file, or multiple files
  -m MAX_SIZE_MB, --max_size_mb MAX_SIZE_MB
                        The maximum file size for each concatenated file in megabytes
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory where the preprocessed data will be saved
  -t, --tokenize        Whether to tokenize the data after preprocessing
  --type DTYPE          Data type for saved token arrays (e.g., int32, uint16, int64). Default: int32
  -n NUM_OF_DATASET, --num_of_dataset NUM_OF_DATASET
                        Maximum number of output dataset files to create (default: None, process all)
  -v, --verify          Verify that token files match the text files in output_dir
  -c, --combine         Combine input files into larger files. If not specified, inferred from input type (False for file lists, True for directories/wildcards)
```

e.g. The command below to combine books into 100M text files and save tokenized text data into npy files.

```
uv run src/mlxgpt/prepare_dataset.py -i gutenberg/data/raw/ -o gutenberg_processed -t -m 100
```

Then pretrain a small GPT-2 model using:
```
uv run src/mlxgpt/train.py -i gutenberg_processed/*.npy -s small -t 1 -p -c -e 100
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
```


```bash
$ uv run src/mlxgpt/gpt2.py -p "To be or not to be"
Loading gpt2-small (124M) model from cache...
Model loaded successfully from gpt2/GPT2-small.npz

Prompt: To be or not to be
Generating 50 tokens with temperature=1.0, top_k=25...

Generated text:
To be or not to be a part of the future. We're going to find something that really is going to bring out the real and the real."


On The Big Chill: "We made it and that will happen in our country, but it's only coming through
```
