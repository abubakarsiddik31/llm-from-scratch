# The BPE Tokenizer

> **Project source:** [`phase1_foundation/project2_tokenizer/`](https://github.com/abubakarsiddik31/llm-from-scratch/tree/main/phase1_foundation/project2_tokenizer)
>
> **Papers:** *"Byte-Pair Encoding: Subword-Based Machine Translation"* (Sennrich, Haddow & Birch, 2016) · *"A New Algorithm for Data Compression"* (Gage, 1994 — original BPE) · WikiText (Merity et al., 2016)

## Why subwords?

Chapter 1's model saw characters — no unknown tokens, but every word cost
many positions of context. Word-level models waste the opposite way: huge
vocabularies and no way to handle unseen words. Byte-Pair Encoding sits in
between, and it is the tokenization used by GPT-2, GPT-3, and most modern
LLMs.

| Approach | Example | Pros | Cons |
|----------|---------|------|------|
| **Character** | `"hello"` → `h e l l o` | No unknowns, tiny vocab | Very long sequences |
| **Word** | `"hello"` → `hello` | Short sequences | Huge vocab, unknowns |
| **BPE** | `"hello"` → `hello`, `"hells"` → `hell · s` | Balanced, handles OOV | Moderate vocab |

Key properties:

1. **Handles OOV**: `"unfriendliness"` → `un · friend · li · ness`
2. **Efficient**: frequent words become single tokens
3. **Compositional**: subwords recur across related words (`help`, `help · ing`, `help · ful`)
4. **No unknowns**: worst case, every text degrades to characters

## The training algorithm

From Sennrich et al. (2016), Algorithm 1:

```
1. Initialize the vocabulary with all unique characters
2. Repeat until the target vocabulary size is reached:
   a. Count the frequency of every adjacent token pair
   b. Find the most frequent pair
   c. Merge it into a new token
   d. Apply the merge everywhere in the corpus
```

Worked example on `"hug hug pug hug pug"`:

| Step | Operation | Vocabulary |
|------|-----------|------------|
| Initial | characters | `{h, u, g, p, ␣}` |
| 1 | merge `u`+`g` → `ug` | `{h, u, g, p, ␣, ug}` |
| 2 | merge `h`+`ug` → `hug` | `{h, u, g, p, ␣, ug, hug}` |
| 3 | merge `p`+`ug` → `pug` | `{h, u, g, p, ␣, ug, hug, pug}` |

After training, encoding is **greedy longest-match-first**: apply the learned
merges in the order they were learned.

```
Learned merges: [('e','r'), ('er','t'), ('ert','a')]
Text: "erter"
Step 1: ['e','r','t','e','r']
Step 2: ['er','t','e','r']      (apply 'e'+'r')
Step 3: ['ert','e','r']         (apply 'er'+'t')
Step 4: ['ert','er']            (apply 'e'+'r')
```

## Scaling the training loop

The naive implementation recounts every adjacent pair across the *entire
corpus* on every merge — O(corpus length) per merge. On an 11 MB WikiText
corpus with a 5,000-token vocabulary that measured **~4.7 s/merge, a 5-hour
run**.

The fix in this repository: represent the corpus as **distinct words with
multiplicities** (a `Counter`), so each merge iterates over ~50k unique words
instead of millions of characters, weighting pair counts by word frequency.
Merges still cross the word/space boundary because every word carries a
trailing space into training (so tokens like `"the "` are learnable).

**Measured: ~5.8 merges/s → the same run finishes in ~12 minutes**, a ~50×
speedup with an identical resulting vocabulary.

## Data and configuration

```bash
# WikiText-2 (~10 MB, good for testing) or WikiText-103 (~500 MB)
uv run python phase1_foundation/project2_tokenizer/download_data.py --size small

# Train (5k vocab trains in minutes; GPT-2 uses 50,257)
uv run python phase1_foundation/project2_tokenizer/train_tokenizer.py --vocab_size 5000

# Round-trip tests + interactive exploration
uv run python phase1_foundation/project2_tokenizer/test_tokenizer.py --interactive
```

## Special tokens

| Token | ID | Purpose |
|-------|----|---------|
| `<PAD>` | 0 | Padding for batching |
| `<UNK>` | 1 | Unknown fallback (rare — BPE covers all bytes) |
| `<BOS>` | 2 | Beginning of sequence |
| `<EOS>` | 3 | End of sequence |

These positional ids matter later: Chapter 3/4's BERT-style models frame
sequences by *position* (0 = pad, 2 = sentence start, 3 = separator).

## What you should observe

- Very common words (`the`, `be`) encode as a single token
- Technical words split meaningfully: `hyper · parameter`, `transform · er`
- Compression of typical English text is ~2.5–3× versus characters

## Code tour

| File | What to read |
|------|--------------|
| [`tokenizer.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project2_tokenizer/tokenizer.py) | `BPETokenizer.train()` (with the distinct-word optimization), `_apply_merge`, greedy `encode()` |
| [`train_tokenizer.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project2_tokenizer/train_tokenizer.py) | CLI, checkpoint format |
| [`test_tokenizer.py`](https://github.com/abubakarsiddik31/llm-from-scratch/blob/main/phase1_foundation/project2_tokenizer/test_tokenizer.py) | Round-trip and behavioral tests |

## Exercises

1. Train with `--vocab_size 1000` vs `5000` and compare token counts for the
   same paragraph. Where does the extra vocabulary pay off?
2. Break the round-trip: find an input whose `decode(encode(x)) != x`, and
   explain why.
3. (Hard) Implement incremental pair-count updates — only pairs adjacent to a
   merge site change — and benchmark against the distinct-word approach.

## What's next

Chapter 3 switches from generation to *understanding*: a bidirectional
encoder pre-trained with masked language modeling, using this tokenizer.
