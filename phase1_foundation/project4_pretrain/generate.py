# ruff: noqa
"""
Text Generation from a Pre-Trained Checkpoint

Loads checkpoints/project4/model_final.pt (or a five-key checkpoint via
--checkpoint), rebuilds the architecture from the config embedded in the
checkpoint, loads the tokenizer, and samples text with the same
temperature / top-k controls as the chapter 1 generate script.

All loads use weights_only=True: the checkpoint format contains tensors
and primitives only, so nothing in the file can execute on load.

USAGE:
------
uv run python phase1_foundation/project4_pretrain/generate.py --prompt "The theory"
uv run python phase1_foundation/project4_pretrain/generate.py --interactive
uv run python phase1_foundation/project4_pretrain/generate.py --checkpoint checkpoint_best.pt
"""

import argparse
import pathlib

import torch

import config
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def load_model(checkpoint_name: str, device: str):
    """
    Rebuild the model from a checkpoint.

    The final save carries a 'config' dict with the exact architecture; the
    five-key interval checkpoints do not, so those fall back to the current
    config module (change config.py between training and generation and
    state-dict loading will fail loudly, as it should).
    """
    path = pathlib.Path(config.CHECKPOINT_DIR) / checkpoint_name
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}. Run train.py first.")

    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = ckpt.get("config")
    if arch is None:
        arch = {
            "VOCAB_SIZE": config.VOCAB_SIZE,
            "N_EMBD": config.N_EMBD,
            "N_HEAD": config.N_HEAD,
            "N_LAYER": config.N_LAYER,
            "BLOCK_SIZE": config.BLOCK_SIZE,
            "DROPOUT": 0.0,
        }
    model = GPT(GPTConfig(
        VOCAB_SIZE=arch["VOCAB_SIZE"],
        N_EMBD=arch["N_EMBD"],
        N_HEAD=arch["N_HEAD"],
        N_LAYER=arch["N_LAYER"],
        BLOCK_SIZE=arch["BLOCK_SIZE"],
        DROPOUT=0.0,
    ))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    val_loss = ckpt.get("val_loss")
    print(f"Loaded {path.name} (iter {ckpt.get('iter')}"
          f"{f', val loss {val_loss:.4f}' if val_loss is not None else ''})")
    return model


def sample(model, tokenizer, prompt: str, args, device: str) -> str:
    """Encode the prompt, sample from it, decode back to text."""
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [config.SPECIAL_TOKENS["<BOS>"]]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0].tolist()
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text from the pre-trained GPT")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt text (empty = start from <BOS>)")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--checkpoint", type=str, default="model_final.pt",
                        help="Checkpoint file name inside checkpoints/project4/")
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--interactive", action="store_true",
                        help="Keep prompting in a loop (Ctrl+C to exit)")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    tokenizer = BPETokenizer.load()

    if args.interactive:
        print("\nInteractive generation (Ctrl+C to exit)")
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                break
            print(sample(model, tokenizer, prompt, args, args.device))
    else:
        print(sample(model, tokenizer, args.prompt, args, args.device))


if __name__ == "__main__":
    main()
