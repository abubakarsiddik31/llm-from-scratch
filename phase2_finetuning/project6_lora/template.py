# ruff: noqa
"""
The Alpaca Prompt Template and the Response-Mask Arithmetic

Every fine-tuning example is an (instruction, input, output) triple rendered into
one plain-text prompt with Alpaca's markers (Taori et al., 2023):

    Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:

The model's job is to produce what follows "### Response:". The <EOS>
special token after the response marks where to stop; generate.py
listens for it when sampling. This file is a verbatim copy of project
5's (the LoRA project trains on the same encoded Alpaca arrays and asks
the same template question at inference time).

WHY PLAIN-TEXT MARKERS (no new special tokens):
-----------------------------------------------
Adding tokens like <|assistant|> would force an embedding-table resize,
which touches the base model's weights. The project 4 tokenizer already
knows every character in "### Instruction:" - it read 100 MB of English
prose - so plain text markers cost nothing and keep "SFT attaches to the
base model unchanged" literally true. Real tokenizer kits (GPT-2's
<|endoftext|>, Llama's chat roles) do reserve special tokens; that
refinement belongs to a later project.

PAPER: "Stanford Alpaca" (Taori et al., 2023), train.py's PROMPT template;
"Self-Instruct" (Wang et al., 2023) for the instruction/input/output
schema Alpaca inherited.

THE TRAILING-SPACE JUNCTION (the subtle part):
----------------------------------------------
Our BPE tokens include trailing spaces ("the " is one token), and the
tokenizer normalizes all whitespace to single spaces. A word therefore
encodes differently depending on whether another word follows it:
"Response:" at the end of a string has no trailing space, but the same
word mid-stream encodes as "Response: ". Splicing token lists by position
(encode prompt, then find where the response starts) breaks at exactly
this junction.

encode_labeled() sidesteps it by splitting BOTH halves into words first
and encoding each word with the same trailing-space convention encode()
uses internally: every prompt word keeps a trailing space (the response
always follows it), and only the final response word drops it. Because
encode() encodes each word independently, the concatenation of those two
lists is EXACTLY encode(prompt_text + " " + response) - test_model.py
verifies that identity - and the response starts at
1 + len(prompt_ids), where the 1 is <BOS.
"""

from typing import List, Tuple

import config

PREAMBLE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
INSTRUCTION_MARKER = "### Instruction:"
INPUT_MARKER = "### Input:"
RESPONSE_MARKER = "### Response:"


def format_prompt(instruction: str, user_input: str = "") -> str:
    """
    Render an (instruction, input) pair into the prompt HALF of an example
    (everything the model conditions on, response marker included).

    The optional "### Input:" block is dropped when the example has no
    input - Alpaca's own template rule; most of the 52k examples skip it.
    """
    parts = [PREAMBLE, f"{INSTRUCTION_MARKER}\n{instruction.strip()}"]
    if user_input.strip():
        parts.append(f"{INPUT_MARKER}\n{user_input.strip()}")
    parts.append(RESPONSE_MARKER)
    return "\n\n".join(parts)


def encode_labeled(tokenizer, instruction: str, user_input: str, output: str,
                   max_seq_len: int = config.MAX_SEQ_LEN) -> Tuple[List[int], List[int]]:
    """
    Encode one example into (tokens, labels) with response-only masking.

    Returns:
        tokens: [<BOS>] + prompt + response + [<EOS>], padded to max_seq_len
                with <PAD> (id 0).
        labels: same length; tokens to TEACH are copied, everything else
                (prompt, pads) is -1, the cross-entropy ignore index used
                by model.forward.

    The caller shifts by one at batch time (x = tokens[:-1],
    y = labels[1:]), so label position t is the target predicted FROM
    input position t-1 - the first response token is predicted from the
    last prompt token, and <EOS> is predicted from the last response
    token. Both are exactly what SFT should teach.

    Raises ValueError when the prompt alone fills the window (no room for
    a response) or the output is empty; callers count and skip those.
    """
    prompt_text = format_prompt(instruction, user_input)
    output = output.strip()
    if not output:
        raise ValueError("empty output")
    if not instruction.strip():
        raise ValueError("empty instruction")

    prompt_ids = tokenizer.encode_words(prompt_text.split(), final=False)
    response_ids = tokenizer.encode_words(output.split(), final=True)

    # +1 for the <BOS> prepended below.
    response_start = len(prompt_ids) + 1
    if response_start + 2 > max_seq_len:  # +2: one response token + <EOS>
        raise ValueError(
            f"prompt needs {response_start} of {max_seq_len} positions; "
            f"no room for a response"
        )

    bos = config.SPECIAL_TOKENS["<BOS>"]
    eos = config.SPECIAL_TOKENS["<EOS>"]
    pad = config.SPECIAL_TOKENS["<PAD>"]

    tokens = [bos] + prompt_ids + response_ids
    if len(tokens) >= max_seq_len:
        # truncate the RESPONSE to fit; the prompt stays intact and <EOS>
        # closes whatever response remains
        tokens = tokens[:max_seq_len - 1] + [eos]
    else:
        tokens.append(eos)
    tokens = tokens + [pad] * (max_seq_len - len(tokens))

    labels = [-1] * len(tokens)
    for i in range(response_start, len(tokens)):
        if tokens[i] != pad:
            labels[i] = tokens[i]
    return tokens, labels
