# ruff: noqa
"""
BERT-Style Contextual Embeddings Implementation

This file implements BERT architecture following:
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
  (Devlin et al., 2018)
- "Attention is All You Need" (Vaswani et al., 2017) - Base transformer architecture

Each component is explained with paper references for educational purposes.

ARCHITECTURE REFERENCE:
-----------------------
From BERT paper (2018):
"BERT is designed to pre-train deep bidirectional representations from unlabeled
text by jointly conditioning on both left and right context in all layers."

KEY DIFFERENCES FROM GPT:
-------------------------
1. BIDIRECTIONAL: No causal mask - can see all tokens
2. ENCODER-ONLY: No generation - produces contextual embeddings
3. MLM TRAINING: Masked Language Modeling (predict masked tokens)
4. NSP TASK: Next Sentence Prediction (binary classification)

From the paper:
"Unlike BERT, GPT uses a unidirectional (left-to-right) architecture...
This restriction is sub-optimal for sentence-level tasks."

ARCHITECTURE:
-------------
Input Token Embeddings
    ↓
Add Positional Embeddings
    ↓
Add Segment Embeddings (for sentence pairs)
    ↓
Transformer Encoder Blocks × N
    ├── Bidirectional Self-Attention (Q, K, V projections + multi-head)
    ├── Add & Norm (Residual + LayerNorm)
    ├── FeedForward (4× expansion)
    └── Add & Norm
    ↓
Two Output Heads:
    1. MLM Head: Predict masked tokens (vocab_size output)
    2. NSP Head: Predict next sentence (binary output)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import config


# =============================================================================
# PART 1: BIDIRECTIONAL SELF-ATTENTION
# =============================================================================


class BidirectionalSelfAttention(nn.Module):
    """
    Bidirectional (Unmasked) Self-Attention Mechanism

    PAPER: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
    ----------------------------------------------------------------------------------

    KEY DIFFERENCE FROM GPT:
    ------------------------
    From BERT paper:
    "BERT uses a bidirectional Transformer... unlike GPT which uses a
    unidirectional (left-to-right) architecture."

    GPT (Causal): Each token can only attend to previous tokens
    BERT (Bidirectional): Each token can attend to ALL tokens

    ATTENTION MECHANISM:
    --------------------
    Scaled Dot-Product Attention (same as GPT):
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    The ONLY difference: NO CAUSAL MASK!

    INTUITION:
    ----------
    For sentence "The cat sat on the mat":

    GPT at position 3 ("sat"):
        - Can attend to: "The", "cat", "sat"
        - CANNOT see: "on", "the", "mat"

    BERT at position 3 ("sat"):
        - Can attend to: ALL tokens
        - Full context from both sides

    WHY BIDIRECTIONAL?
    ------------------
    From BERT paper:
    "Limiting unidirectional models is sub-optimal for sentence-level tasks...

    BERT uses... masked language modeling (MLM) objective to enable
    bidirectional pre-training."

    Benefits:
    1. Better sentence understanding: See full context at each position
    2. Better for classification: Use all information
    3. Better for NER: Use both left and right context
    4. Better for SQuAD: Use entire question/passage

    TRADE-OFF:
    -----------
    BERT cannot generate text autoregressively!
    - No causal mask = cannot predict next token
    - Designed for understanding, not generation
    - For generation, use GPT-style causal attention
    """

    def __init__(self, config):
        super().__init__()

        # ===========================================================================
        # VALIDATION
        # ===========================================================================

        """
        From BERT paper: "For BERT-base, we use 12 attention heads...
        with d_model = 768, so each head has dimension 64"

        n_embd must be divisible by n_head.
        """
        assert config.N_EMBD % config.N_HEAD == 0
        self.n_head = config.N_HEAD
        self.n_embd = config.N_EMBD
        self.head_size = config.N_EMBD // config.N_HEAD

        # ===========================================================================
        # Q, K, V PROJECTIONS
        # ===========================================================================

        """
        Same as GPT: Combined projection for efficiency.
        Single matrix multiply produces all three.
        """
        self.c_attn = nn.Linear(config.N_EMBD, 3 * config.N_EMBD, bias=False)

        # ===========================================================================
        # OUTPUT PROJECTION
        # ===========================================================================

        """
        After concatenating heads, project back to n_embd.
        """
        self.c_proj = nn.Linear(config.N_EMBD, config.N_EMBD, bias=False)

        # ===========================================================================
        # REGULARIZATION
        # ===========================================================================

        """
        BERT uses dropout on attention weights (same as GPT).
        """
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.resid_dropout = nn.Dropout(config.DROPOUT)

        # NOTE: No causal mask! That's the key difference from GPT.

    def forward(self, x):
        """
        Forward pass of bidirectional self-attention.

        PAPER REFERENCE: "Attention is All You Need" (Section 3.2.1)

        COMPUTATION STEPS:
        ------------------
        1. Compute Q, K, V: Linear projections from input
        2. Reshape for multi-head: Split across n_head
        3. Compute attention scores: Q @ K^T / √d_k
        4. NO MASKING! (Key difference from GPT)
        5. Softmax: Convert scores to probabilities
        6. Weight values: Attention weights @ V
        7. Concatenate heads: Merge all heads
        8. Output projection: Linear transformation

        Args:
            x: Input tensor of shape (B, T, C)
               B = batch size
               T = sequence length (≤ BLOCK_SIZE)
               C = embedding dimension (N_EMBD)

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()

        # ===========================================================================
        # 1. COMPUTE Q, K, V
        # ===========================================================================

        """
        Same as GPT: Single matrix multiplication.
        """
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # ===========================================================================
        # 2. RESHAPE FOR MULTI-HEAD
        # ===========================================================================

        """
        Same as GPT: Split embedding dimension across heads.
        """
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # ===========================================================================
        # 3. COMPUTE ATTENTION SCORES
        # ===========================================================================

        """
        Same as GPT: Scaled dot-product attention.
        """
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)

        # ===========================================================================
        # 4. NO MASKING! (Key difference from GPT)
        # ===========================================================================

        """
        GPT: Apply causal mask (future positions = -inf)
        BERT: NO MASK! All positions can attend to all positions

        This is why BERT is "bidirectional" - full context at each position.
        """

        # ===========================================================================
        # 5. SOFTMAX TO GET ATTENTION WEIGHTS
        # ===========================================================================

        """
        Same as GPT: Convert scores to probabilities.
        """
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # ===========================================================================
        # 6. WEIGHT VALUES BY ATTENTION WEIGHTS
        # ===========================================================================

        """
        Same as GPT: Weighted sum of values.
        """
        y = att @ v

        # ===========================================================================
        # 7. CONCATENATE HEADS
        # ===========================================================================

        """
        Same as GPT: Merge all heads.
        """
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # ===========================================================================
        # 8. OUTPUT PROJECTION
        # ===========================================================================

        """
        Same as GPT: Mix information from all heads.
        """
        y = self.resid_dropout(self.c_proj(y))

        return y


# =============================================================================
# PART 2: FEED-FORWARD NETWORK
# =============================================================================


class FeedForward(nn.Module):
    """
    Feed-Forward Network (Position-wise)

    PAPER: "Attention is All You Need" (Section 3.3)
    BERT: Same architecture as GPT

    Same implementation as Project 1 (GPT).
    Applied identically to each position.
    """

    def __init__(self, config):
        super().__init__()

        """
        From BERT paper:
        "We use a feed-forward network with 4× hidden size"
        """
        self.net = nn.Sequential(
            nn.Linear(config.N_EMBD, 4 * config.N_EMBD),
            nn.GELU(),  # BERT uses GELU (smoother than ReLU)
            nn.Linear(4 * config.N_EMBD, config.N_EMBD),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# PART 3: TRANSFORMER ENCODER BLOCK
# =============================================================================


class TransformerEncoderBlock(nn.Module):
    """
    BERT Transformer Encoder Block

    PAPER: "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
    ---------------------------------------------------------------------

    From BERT paper:
    "We use the Transformer encoder architecture... with multi-head
    self-attention over each token."

    ARCHITECTURE:
    -------------
    Pre-LN (GPT-2 style):
        x = x + SelfAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    KEY DIFFERENCE FROM GPT:
    ------------------------
    GPT uses CausalSelfAttention (masked)
    BERT uses BidirectionalSelfAttention (unmasked)

    That's the ONLY structural difference!
    """

    def __init__(self, config):
        super().__init__()

        # ===========================================================================
        # ATTENTION AND FFN
        # ===========================================================================

        """
        BERT uses bidirectional attention instead of causal attention.
        """
        self.attn = BidirectionalSelfAttention(config)
        self.ffwd = FeedForward(config)

        # ===========================================================================
        # LAYER NORMALIZATION
        # ===========================================================================

        """
        Same as GPT: LayerNorm before each sublayer (Pre-LN).
        """
        self.ln1 = nn.LayerNorm(config.N_EMBD)
        self.ln2 = nn.LayerNorm(config.N_EMBD)

    def forward(self, x):
        """
        Forward pass with residual connections.

        Same structure as GPT, just with bidirectional attention.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C)
        """
        # Self-attention with residual
        x = x + self.attn(self.ln1(x))

        # Feed-forward with residual
        x = x + self.ffwd(self.ln2(x))

        return x


# =============================================================================
# PART 4: COMPLETE BERT MODEL
# =============================================================================


class BERT(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers

    PAPERS:
    -------
    1. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
       (Devlin et al., 2018)

    2. "Attention is All You Need" (Vaswani et al., 2017)

    ARCHITECTURE OVERVIEW:
    ----------------------
    From BERT paper:
    "We use the Transformer encoder... with multi-head self-attention"

    Full architecture:
    1. Token Embedding: Convert indices to vectors
    2. Positional Embedding: Add position information
    3. Segment Embedding: Distinguish sentence pairs (A vs B)
    4. Transformer Encoder Blocks: Stack of N blocks (bidirectional)
    5. Output Heads:
       - MLM Head: Predict masked tokens
       - NSP Head: Predict next sentence

    SEGMENT EMBEDDINGS:
    -------------------
    From BERT paper:
    "To handle sentence pairs, we learn a special embedding... segment embedding"

    [CLS] Sentence A [SEP] Sentence B [SEP]
    A A A ... A A B B ... B B

    - Sentence A tokens get segment_id = 0
    - Sentence B tokens get segment_id = 1

    This lets the model distinguish which sentence each token belongs to.

    SPECIAL TOKENS:
    ---------------
    [CLS]: Classification token at start of every sequence
           - Output at this position is used for classification tasks
    [SEP]: Separator token between sentences
    [MASK]: Mask token for MLM training
    [PAD]: Padding token for batching

    PRE-TRAINING TASKS:
    -------------------
    1. MLM (Masked Language Modeling): Predict masked tokens
       - Randomly mask 15% of tokens
       - Predict original tokens

    2. NSP (Next Sentence Prediction): Binary classification
       - Given two sentences, predict if B follows A
       - 50% positive pairs, 50% negative pairs

    FINE-TUNING:
    ------------
    After pre-training, BERT can be fine-tuned for:
    - Classification: Use [CLS] output
    - NER: Use token-level outputs
    - QA: Use span prediction on outputs
    - etc.
    """

    def __init__(self, config):
        super().__init__()

        # Store config for saving/loading
        self.config = config

        # ===========================================================================
        # EMBEDDINGS
        # ===========================================================================

        """
        From BERT paper:
        "The input representation is constructed by summing the token,
        position, and segment embeddings."

        THREE EMBEDDINGS (unlike GPT which has two):
        1. Token embeddings: What word is this?
        2. Position embeddings: Where is this in the sequence?
        3. Segment embeddings: Which sentence does this belong to? (A or B)
        """
        self.embeddings = nn.ModuleDict(
            dict(
                # Token embeddings (same as GPT)
                token=nn.Embedding(config.VOCAB_SIZE, config.N_EMBD),
                # Position embeddings (same as GPT)
                position=nn.Embedding(config.BLOCK_SIZE, config.N_EMBD),
                # Segment embeddings (unique to BERT!)
                # segment_type = 0 for sentence A, = 1 for sentence B
                segment=nn.Embedding(2, config.N_EMBD),
                # Dropout on combined embeddings
                drop=nn.Dropout(config.DROPOUT),
            )
        )

        # ===========================================================================
        # TRANSFORMER ENCODER BLOCKS
        # ===========================================================================

        """
        From BERT paper:
        "BERT-base: 12 layers (Transformer blocks)
         BERT-large: 24 layers"

        Stack of N transformer encoder blocks.
        Each block: bidirectional attention + FFN with residuals.
        """
        blocks = [TransformerEncoderBlock(config) for _ in range(config.N_LAYER)]
        self.blocks = nn.ModuleList(blocks)

        # ===========================================================================
        # OUTPUT LAYER NORM
        # ===========================================================================

        """
        Final LayerNorm before output heads.
        """
        self.ln_f = nn.LayerNorm(config.N_EMBD)

        # ===========================================================================
        # MLM HEAD (Masked Language Modeling)
        # ===========================================================================

        """
        From BERT paper:
        "For the masked language modeling loss... we predict the masked tokens"

        MLM HEAD:
        - Project from hidden state to vocabulary size
        - Same as token embedding matrix (weight tying)
        - Output: logits for each token in vocabulary
        """
        self.mlm_head = nn.Linear(config.N_EMBD, config.VOCAB_SIZE, bias=False)

        # Weight tying: Share embeddings with MLM head
        # From "Using the Output Embedding to Improve Language Models"
        self.mlm_head.weight = self.embeddings.token.weight

        # ===========================================================================
        # NSP HEAD (Next Sentence Prediction)
        # ===========================================================================

        """
        From BERT paper:
        "For next sentence prediction... we add a binary classification head"

        NSP HEAD:
        - Use [CLS] token output (first position)
        - Linear layer: hidden_size → 2
        - Output: IsNext (1) or NotNext (0)

        NOTE: Recent work suggests NSP may not be necessary.
        We include it for educational purposes.
        """
        if config.USE_NSP:
            self.nsp_head = nn.Linear(config.N_EMBD, 2)

        # ===========================================================================
        # WEIGHT INITIALIZATION
        # ===========================================================================

        """
        From BERT paper:
        "We use a truncated normal distribution with mean 0 and std 0.02"

        Similar to GPT-2 initialization.
        """
        self.apply(self._init_weights)

        print(f"BERT model initialized with {self.get_num_params():,} parameters")

    def _init_weights(self, module):
        """
        Initialize weights following BERT conventions.

        From BERT paper and codebase:
        "All parameters are initialized with a truncated normal distribution
        with mean 0 and standard deviation 0.02"
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If False, count embedding params too
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude token embeddings (lookup table)
            n_params -= self.embeddings.token.weight.numel()
        return n_params

    def forward(self, idx, segment_ids=None, masked_labels=None, nsp_labels=None):
        """
        Forward pass of BERT model.

        PAPER: "BERT: Pre-training of Deep Bidirectional Transformers"

        FORWARD PASS:
        1. Lookup token embeddings
        2. Add positional embeddings
        3. Add segment embeddings
        4. Pass through transformer encoder blocks
        5. Final layer norm
        6. Compute MLM logits (for all positions)
        7. Compute NSP logits (using [CLS] token)

        Args:
            idx: Input token indices of shape (B, T)
            segment_ids: Segment indices of shape (B, T)
                         0 for sentence A, 1 for sentence B
            masked_labels: Target token indices for MLM of shape (B, T)
                          Only compute MLM loss for masked positions
            nsp_labels: Binary labels for NSP of shape (B,)
                        1 = IsNext, 0 = NotNext

        Returns:
            mlm_logits: MLM output logits of shape (B, T, vocab_size)
            nsp_logits: NSP output logits of shape (B, 2) (if USE_NSP)
            mlm_loss: MLM cross-entropy loss (scalar)
            nsp_loss: NSP cross-entropy loss (scalar, if USE_NSP)
        """
        device = idx.device
        B, T = idx.shape

        # Validate sequence length
        assert (
            T <= self.config.BLOCK_SIZE
        ), f"Sequence length {T} exceeds block size {self.config.BLOCK_SIZE}"

        # ===========================================================================
        # 1. TOKEN EMBEDDINGS
        # ===========================================================================

        """
        Lookup: Each token index gets its embedding vector.
        """
        tok_emb = self.embeddings.token(idx)

        # ===========================================================================
        # 2. POSITIONAL EMBEDDINGS
        # ===========================================================================

        """
        Create position indices: [0, 1, 2, ..., T-1]
        """
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.embeddings.position(pos)

        # ===========================================================================
        # 3. SEGMENT EMBEDDINGS
        # ===========================================================================

        """
        From BERT paper:
        "To handle sentence pairs, we add a segment embedding"

        If segment_ids is None, assume all tokens belong to sentence A (segment 0).
        """
        if segment_ids is None:
            segment_ids = torch.zeros_like(idx)
        seg_emb = self.embeddings.segment(segment_ids)

        # ===========================================================================
        # 4. COMBINE EMBEDDINGS
        # ===========================================================================

        """
        From BERT paper:
        "The input representation is constructed by summing the token,
        position, and segment embeddings."

        x = token_emb + pos_emb + seg_emb
        """
        x = self.embeddings.drop(tok_emb + pos_emb + seg_emb)

        # ===========================================================================
        # 5. TRANSFORMER ENCODER BLOCKS
        # ===========================================================================

        """
        Pass through each encoder block sequentially.
        Bidirectional attention means full context at each position.
        """
        for block in self.blocks:
            x = block(x)

        # ===========================================================================
        # 6. FINAL LAYER NORM
        # ===========================================================================

        """
        Normalize before output heads.
        """
        x = self.ln_f(x)

        # ===========================================================================
        # 7. MLM LOGITS
        # ===========================================================================

        """
        Project hidden states to vocabulary for MLM prediction.
        Output shape: (B, T, vocab_size)
        Each position has predictions for all vocabulary tokens.
        """
        mlm_logits = self.mlm_head(x)

        # ===========================================================================
        # 8. NSP LOGITS (if enabled)
        # ===========================================================================

        """
        From BERT paper:
        "For the NSP task, we use the output of the [CLS] token"

        [CLS] is at position 0 (first token in sequence).
        """
        nsp_logits = None
        if self.config.USE_NSP:
            # Use [CLS] token output (position 0)
            cls_output = x[:, 0, :]  # Shape: (B, hidden_size)
            nsp_logits = self.nsp_head(cls_output)  # Shape: (B, 2)

        # ===========================================================================
        # 9. LOSS COMPUTATION
        # ===========================================================================

        mlm_loss = None
        nsp_loss = None

        if masked_labels is not None:
            """
            MLM Loss: Cross-entropy for masked positions only.

            From BERT paper:
            "We only predict the masked tokens and ignore the non-masked tokens"

            We use -100 as ignore_index (PyTorch convention).
            """
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                masked_labels.view(-1),
                ignore_index=-100,
            )

        if nsp_labels is not None and self.config.USE_NSP:
            """
            NSP Loss: Binary cross-entropy for next sentence prediction.
            """
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)

        return mlm_logits, nsp_logits, mlm_loss, nsp_loss


# =============================================================================
# PART 5: UTILITY FUNCTIONS
# =============================================================================


def create_bert_model(vocab_size, config=config):
    """
    Helper function to create BERT model with specified vocabulary size.

    Args:
        vocab_size: Size of the vocabulary
        config: Configuration object

    Returns:
        Initialized BERT model
    """
    # Set vocab size in config
    config.VOCAB_SIZE = vocab_size

    # Create model
    model = BERT(config)

    return model


if __name__ == "__main__":
    # Quick self-test
    print("BERT Model - Self Test")
    print("=" * 60)

    # Set a small vocab size for testing
    test_vocab_size = 100
    config.VOCAB_SIZE = test_vocab_size

    # Create model
    model = BERT(config)

    # Test forward pass
    B, T = 2, 16
    idx = torch.randint(0, test_vocab_size, (B, T))
    segment_ids = torch.zeros(B, T, dtype=torch.long)

    print(f"\nInput shape: {idx.shape}")

    # Forward pass
    mlm_logits, nsp_logits, mlm_loss, nsp_loss = model(idx, segment_ids)

    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"NSP logits shape: {nsp_logits.shape if nsp_logits is not None else 'N/A'}")

    # Test with labels
    masked_labels = torch.randint(0, test_vocab_size, (B, T))
    masked_labels[:, :8] = -100  # Mask first half as non-masked

    nsp_labels = torch.randint(0, 2, (B,))

    mlm_logits, nsp_logits, mlm_loss, nsp_loss = model(
        idx, segment_ids, masked_labels, nsp_labels
    )

    print(f"\nWith labels:")
    print(f"MLM loss: {mlm_loss.item():.4f}")
    print(f"NSP loss: {nsp_loss.item():.4f}" if nsp_loss is not None else "NSP loss: N/A")

    print("\n✓ Self-test passed!")
