# Project 1: Usage Guide

## Quick Start

```bash
# 1. Download data
cd week1/project1_minimal_gpt
python download_data.py

# 2. Train the model
python train.py

# 3. Generate text
python generate.py --prompt "ROMEO:" --interactive

# Or generate once
python generate.py --prompt "HAMLET:" --max_tokens 200
```

## Understanding the Output

### During Training

```
Step 0:
  Train loss: 4.50    # Random predictions (high)
  Val loss:   4.50
  Sample:    FjbKpLz...  # Garbage

Step 500:
  Train loss: 2.80    # Learning patterns
  Val loss:   2.82
  Sample:    The Duke of Venice...  # Some structure

Step 2000:
  Train loss: 1.90    # Coherent text
  Val loss:   1.92
  Sample:    ROMEO:
              What lady is that that doth enrich the hand...

Step 5000:
  Train loss: 1.60    # Shakespeare-like!
  Val loss:   1.65
  Sample:    ROMEO:
              O, she doth teach the torches to burn bright!
              It seems she hangs upon the cheek of night...
```

### Loss Interpretation

| Loss | Meaning |
|------|---------|
| > 3.0 | Model is barely better than random |
| 2.0-3.0 | Model learned basic structure |
| 1.5-2.0 | Model generates coherent text |
| < 1.5 | Model captures style well |

## Generation Parameters

### Temperature

```bash
# Conservative (less random)
python generate.py --prompt "ROMEO:" --temperature 0.3

# Balanced (default)
python generate.py --prompt "ROMEO:" --temperature 0.8

# Creative (more random)
python generate.py --prompt "ROMEO:" --temperature 1.5
```

### Top-K Sampling

```bash
# Only sample from top 30 tokens (more focused)
python generate.py --prompt "ROMEO:" --top_k 30

# Only sample from top 50 tokens (default)
python generate.py --prompt "ROMEO:" --top_k 50

# Disable top-k (sample from all tokens)
python generate.py --prompt "ROMEO:" --top_k 0
```

## Interactive Mode

```bash
python generate.py --interactive
```

In interactive mode:
- Type any text to generate continuation
- `temp 0.5` - Change temperature
- `topk 40` - Change top-k value
- `clear` - Clear screen
- `quit` - Exit

## Experiment Ideas

1. **Change context length** in `config.py`:
   ```python
   BLOCK_SIZE = 512  # More context, slower training
   ```

2. **Change model size** in `config.py`:
   ```python
   N_LAYER = 12      # Deeper model
   N_EMBD = 768      # Wider model
   ```

3. **Train on your own text**:
   ```bash
   # Put your text in data/custom.txt
   # Modify train.py to use it
   ```

## Troubleshooting

### Out of Memory

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 32  # or 16
```

### Training Too Slow

- Use a smaller model: `N_LAYER = 4`, `N_EMBD = 256`
- Use shorter context: `BLOCK_SIZE = 128`
- Reduce eval frequency: `EVAL_INTERVAL = 1000`

### Generated Text is Repetitive

- Increase `temperature` above 1.0
- Reduce `top_k` to focus on better tokens
- Train longer (increase `MAX_ITERS`)

## Checkpoints

Checkpoints are saved to `checkpoints/project1/`:
- `checkpoint_iter{N}.pt` - Best models at each evaluation
- `model_final.pt` - Final model after training
- `meta.pkl` - Vocabulary mappings (encode/decode)

## Next Steps

After Project 1, you'll understand:
- ✓ Self-attention mechanism
- ✓ Transformer architecture
- ✓ Training language models
- ✓ Text generation strategies

Ready for **Project 2: BPE Tokenizer**!
