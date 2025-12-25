"""
Text Generation Script for Character-Level GPT

This script loads a trained model and generates text.

Key concepts explained:
- Loading trained models from checkpoints
- Temperature-controlled sampling
- Top-k sampling for better quality
- Interactive generation mode
"""

import os
import torch
import pickle
import argparse

import config
from model import GPT


# =============================================================================
# PART 1: MODEL LOADING
# =============================================================================


def load_model(checkpoint_path, meta_path):
    """
    Load a trained model and metadata.

    CHECKPOINT STRUCTURE:
    -------------------
    The checkpoint file contains:
    - model_state_dict: Learned parameters
    - config: Model configuration
    - optimizer_state_dict: Optimizer state (for resuming training)
    - train_loss, val_loss: Training metrics

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        meta_path: Path to metadata file (contains encode/decode functions)

    Returns:
        model: Loaded GPT model
        encode: Function to encode text to indices
        decode: Function to decode indices to text
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # Load model state
    model_state = checkpoint["model_state_dict"]

    # Reconstruct model (we need the config first)
    # For simplicity, we'll use the current config
    # In production, save/load config from checkpoint
    model = GPT(config)
    model.load_state_dict(model_state)
    model.to(config.DEVICE)
    model.eval()  # Set to evaluation mode (disables dropout)

    # Load metadata (encode/decode functions)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    encode = meta["encode"]
    decode = meta["decode"]

    print(f"✓ Model loaded")

    # Print training info if available
    if "train_loss" in checkpoint:
        print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    if "val_loss" in checkpoint:
        print(f"  Val loss:   {checkpoint['val_loss']:.4f}")
    if "iter" in checkpoint:
        print(f"  Iteration:  {checkpoint['iter']}")

    return model, encode, decode


# =============================================================================
# PART 2: GENERATION
# =============================================================================


def generate_text(
    model, prompt, decode, max_new_tokens=500, temperature=0.8, top_k=None
):
    """
    Generate text given a prompt.

    GENERATION PARAMETERS:
    --------------------

    temperature:
    - Controls randomness of sampling
    - < 1.0: More conservative, picks high-probability tokens
    - = 1.0: Standard sampling from predicted distribution
    - > 1.0: More diverse, surprising outputs
    - Very low (~0.1): Almost deterministic

    top_k:
    - Only sample from the k most likely tokens
    - None: Sample from all tokens
    - Lower value: More focused, less random
    - Typical values: 40, 50

    EXAMPLES:
    --------
    temperature=0.8, top_k=50: Balanced, coherent text
    temperature=0.3, top_k=30: Very conservative, repetitive
    temperature=1.5, top_k=None: Very creative, possibly nonsensical

    Args:
        model: Trained GPT model
        prompt: Starting text (string)
        decode: Function to decode indices to text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling threshold

    Returns:
        Generated text (string)
    """
    # Encode prompt
    # If prompt is empty, start with a single newline (common in training)
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=config.DEVICE)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)

    print(f"\nGenerating {max_new_tokens} tokens...")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("-" * 60)

    # Generate
    generated = model.generate(
        context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )

    # Decode to text
    text = decode(generated[0].tolist())

    return text


# =============================================================================
# PART 3: INTERACTIVE MODE
# =============================================================================


def interactive_mode(model, encode, decode):
    """
    Interactive mode for continuous generation.

    Commands:
    - Type text and press Enter to generate continuation
    - 'quit' or 'exit' to quit
    - 'temp <value>' to change temperature
    - 'topk <value>' to change top-k
    - 'clear' to clear screen
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Commands:")
    print("  Type text → Generate continuation")
    print("  'quit' or 'exit' → Quit")
    print("  'temp <0.1-2.0>' → Set temperature")
    print("  'topk <value>' → Set top-k sampling")
    print("  'clear' → Clear screen")
    print("=" * 60)

    temperature = 0.8
    top_k = 50

    while True:
        try:
            # Get user input
            prompt = input(f"\n[temp={temperature}, topk={top_k}] Prompt: ")

            # Handle commands
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif prompt.lower() == "clear":
                os.system("clear" if os.name == "posix" else "cls")
                continue
            elif prompt.lower().startswith("temp "):
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            elif prompt.lower().startswith("topk "):
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
                continue

            # Generate
            text = generate_text(
                model,
                prompt,
                decode,
                max_new_tokens=500,
                temperature=temperature,
                top_k=top_k,
            )

            print("\nGenerated:")
            print(text)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with new prompt.")
        except Exception as e:
            print(f"\nError: {e}")


# =============================================================================
# PART 4: MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained GPT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/project1/model_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="checkpoints/project1/meta.pkl",
        help="Path to metadata file",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Starting prompt for generation"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature (0.1-2.0)"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train the model first using train.py")
        exit(1)

    if not os.path.exists(args.meta):
        print(f"Metadata not found: {args.meta}")
        exit(1)

    # Load model
    config.validate_config()
    model, encode, decode = load_model(args.checkpoint, args.meta)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, encode, decode)
    else:
        # Single generation
        if args.top_k == 0:
            args.top_k = None

        text = generate_text(
            model,
            args.prompt,
            decode,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        print("\nGenerated:")
        print(text)
