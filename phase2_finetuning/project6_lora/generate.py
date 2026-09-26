# ruff: noqa
"""
Instruction Generation from a LoRA Checkpoint

Loads checkpoints/project6/model_final.pt (or another checkpoint via
--checkpoint), rebuilds the architecture from the embedded config, applies
LoRA according to the embedded lora_config, loads the wrapped state dict,
and answers instructions with the Alpaca template.

The sampler is the same <EOS>-listening loop as project 5's generate.py
(see that file for the full why: model.generate() has no stop condition,
and the SFT-taught <EOS> token is what ends a turn). The prompt is
encoded word-by-word with the trailing-space convention training used.

Checkpoint compatibility rules:
- a project 6 checkpoint (has 'lora_config'): apply LoRA first, then load
  the state dict strictly - the wrapper recreates the exact keys.
- a plain checkpoint (no 'lora_config', e.g. project 5's model_final.pt):
  loads as-is, so this script can also sample the full-SFT model for the
  chapter's comparison.

All loads use weights_only=True.

USAGE:
------
uv run python phase2_finetuning/project6_lora/generate.py --instruction "Give three tips for staying healthy."
uv run python phase2_finetuning/project6_lora/generate.py --interactive
uv run python phase2_finetuning/project6_lora/generate.py --checkpoint checkpoint_best.pt --instruction "What is 2+2?"
"""

import argparse
import pathlib
from typing import List

import torch

import config
import lora
from model import GPT, GPTConfig
from template import format_prompt
from tokenizer import BPETokenizer


def load_model(checkpoint_name: str, device: str):
    """
    Rebuild the model from a checkpoint, wrapping with LoRA when the
    checkpoint says it was saved with adapters.

    Falls back to the local config module when a checkpoint lacks the
    embedded 'config' dict; state-dict loading then fails loudly on any
    mismatch, as it should.
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

    lora_cfg = ckpt.get("lora_config")
    if lora_cfg is not None:
        lora.apply_lora(model, lora_cfg["targets"], lora_cfg["r"], lora_cfg["alpha"])
        trainable, total = lora.lora_parameter_report(model)
        print(f"LoRA applied: r={lora_cfg['r']} alpha={lora_cfg['alpha']} "
              f"targets={lora_cfg['targets']} "
              f"({trainable:,} trainable = {trainable / total:.3%})")
    else:
        print("Plain checkpoint (no lora_config): loading without adapters")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    val_loss = ckpt.get("val_loss")
    base = ckpt.get("base_checkpoint")
    print(f"Loaded {path.name} (iter {ckpt.get('iter')}"
          f"{f', val loss {val_loss:.4f}' if val_loss is not None else ''}"
          f"{f', base {base}' if base else ''})")
    return model


@torch.no_grad()
def sample_response(model, tokenizer, prompt_text: str,
                    max_new_tokens: int = config.MAX_NEW_TOKENS,
                    temperature: float = config.TEMPERATURE,
                    top_k: int = config.TOP_K,
                    device: str = config.DEVICE) -> str:
    """
    Template-conditioned sampling with an <EOS> stop.

    Same algorithm as project 5's generate.py (temperature / top-k, Fan et
    al., 2018; word-by-word prompt encoding; stop when the model draws
    <EOS>). Duplicated here rather than imported so each project stays
    self-contained.
    """
    prompt_ids = tokenizer.encode_words(prompt_text.split(), final=False)
    if not prompt_ids:
        prompt_ids = [config.SPECIAL_TOKENS["<BOS>"]]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos = config.SPECIAL_TOKENS["<EOS>"]

    generated: List[int] = []
    for _ in range(max_new_tokens):
        idx_crop = (
            idx if idx.size(1) <= model.config.BLOCK_SIZE
            else idx[:, -model.config.BLOCK_SIZE:]
        )
        logits, _ = model(idx_crop)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1))
        if next_id == eos:
            break
        generated.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)

    return tokenizer.decode(generated)


def answer(model, tokenizer, instruction: str, user_input: str, args) -> str:
    """Format the template, (optionally) show it, sample the response."""
    prompt = format_prompt(instruction, user_input)
    if getattr(args, "show_prompt", False):
        print("--- prompt sent to the model ---")
        print(prompt)
        print("--- response ---")
    return sample_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Answer instructions with the LoRA-fine-tuned GPT")
    parser.add_argument("--instruction", type=str, default="Give three tips for staying healthy.")
    parser.add_argument("--input", type=str, default="",
                        help="Optional '### Input:' context for the instruction")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--checkpoint", type=str, default="model_final.pt",
                        help="Checkpoint file name inside checkpoints/project6/")
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--show_prompt", action="store_true",
                        help="Print the exact prompt before sampling")
    parser.add_argument("--interactive", action="store_true",
                        help="Ask instructions in a loop (Ctrl+C to exit)")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    tokenizer = BPETokenizer.load()

    if args.interactive:
        print("\nInteractive instruction answering (Ctrl+C to exit)")
        print("Enter an instruction; a second line (optional) becomes '### Input:'.")
        while True:
            try:
                instruction = input("\ninstruction> ").strip()
                if not instruction:
                    continue
                user_input = input("input (Enter to skip)> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                break
            print(answer(model, tokenizer, instruction, user_input, args))
    else:
        print(answer(model, tokenizer, args.instruction, args.input, args))


if __name__ == "__main__":
    main()
