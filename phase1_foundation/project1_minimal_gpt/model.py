# ruff: noqa
"""
Character-Level GPT Model Implementation

This file implements the complete GPT architecture following:
- "Attention is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

Each component is explained with paper references for educational purposes.

ARCHITECTURE REFERENCE:
---------------------
From GPT-1 (2018):
"We use a 12-layer decoder-only transformer with masked self-attention heads...

Key architectural decisions from papers:
1. Decoder-only: No encoder, just causal self-attention
2. Pre-LayerNorm: LayerNorm before sublayers (more stable for deep nets)
3. Learned positional embeddings (not sinusoidal from original Transformer)
4. 4× expansion in FFN (standard from GPT-2/GPT-3)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import configuration
import config


# =============================================================================
# PART 1: CAUSAL SELF-ATTENTION
# =============================================================================


class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention Mechanism

    PAPER: "Attention is All You Need" (Vaswani et al., 2017)
    ------------------------------------------------------------

    THE CORE ATTENTION MECHANISM:
    ----------------------------
    Scaled Dot-Product Attention (Section 3.2.1):

        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Where:
    - Q (Query): "What information am I looking for?"
    - K (Key): "What information do I contain?"
    - V (Value): "What is my actual content?"

    INTUITION:
    ----------
    Think of attention as a database query:
    - Query: Your search query
    - Key: Database indices
    - Value: Database values
    - Attention: Weighted sum of values based on key-query match

    MULTI-HEAD ATTENTION (Section 3.2.2):
    ----------------------------------
    Instead of one attention, use h parallel attentions:

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

        Where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Why multiple heads?
    - Each head learns different relationship patterns
    - Head 1 might learn: subject-verb agreement
    - Head 2 might learn: pronoun reference resolution
    - Head 3 might learn: long-range dependencies
    - etc.

    From the paper: "Allows the model to jointly attend to information
    from different representation subspaces at different positions."

    CAUSAL MASKING:
    ---------------
    From GPT papers: "We use a masked self-attention layer where each
    token can only attend to previous tokens."

    This is implemented with a triangular mask:
    ```
    Position 0 can attend to: [0]
    Position 1 can attend to: [0, 1]
    Position 2 can attend to: [0, 1, 2]
    ...
    Position T can attend to: [0, 1, 2, ..., T]
    ```

    The mask sets future positions to -∞ before softmax, making
    their attention weights zero after softmax.

    WHY "CAUSAL"?
    -------------
    In causal (autoregressive) models, the prediction at position t
    can ONLY depend on positions < t. This ensures:
    1. No "peeking" at future tokens during training
    2. Proper generation: we only have previous tokens when predicting
    3. Consistency between training and inference

    IMPLEMENTATION DETAILS:
    ----------------------
    From GPT-2: "We use masked self-attention layers... with 8 heads."

    Our implementation:
    - Q, K, V projections combined into single matrix (efficiency)
    - bias=False: No learnable bias (standard for attention)
    - Reshape and transpose for multi-head
    - Causal mask: Lower triangular matrix of ones
    """

    def __init__(self, config):
        super().__init__()

        # ===========================================================================
        # VALIDATION
        # ===========================================================================

        """
        From "Attention is All You Need": "We employ h = 8 parallel
        attention layers, or heads. For each of these we use d_k = d_v = d_model/h."

        This means: n_embd must be divisible by n_head
        - Each head gets: n_embd / n_head dimensions
        - Total across all heads: n_head × (n_embd / n_head) = n_embd
        """
        assert config.N_EMBD % config.N_HEAD == 0
        self.n_head = config.N_HEAD
        self.n_embd = config.N_EMBD

        # Head size: dimension per attention head
        # From GPT-2: For n_embd=768, n_head=12, head_size=64
        self.head_size = config.N_EMBD // config.N_HEAD

        # ===========================================================================
        # Q, K, V PROJECTIONS
        # ===========================================================================

        """
        From "Attention is All You Need" (Section 3.2.2):

        "For each of these we use d_k = d_v = d_model/h...
        The queries, keys, and values are linearly projected..."

        Implementation:
        - Single matrix multiplication for all three (efficiency)
        - Output: 3 × n_embd values (n_embd each for Q, K, V)
        - bias=False: No bias term (standard for self-attention)

        Why combined?
        - More efficient: One matrix multiplication instead of three
        - Better memory locality
        - Same result (linear operations are independent)
        """
        self.c_attn = nn.Linear(config.N_EMBD, 3 * config.N_EMBD, bias=False)

        # ===========================================================================
        # OUTPUT PROJECTION
        # ===========================================================================

        """
        From "Attention is All You Need" (Section 3.2.2):

        "The outputs are concatenated and once again projected...

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O"

        After concatenating heads, project back to n_embd dimension.
        This mixes information from all heads.

        bias=False: Standard for attention (learned bias not needed)
        """
        self.c_proj = nn.Linear(config.N_EMBD, config.N_EMBD, bias=False)

        # ===========================================================================
        # REGULARIZATION
        # ===========================================================================

        """
        DROPOUT:
        From "Attention is All You Need":
        "We apply dropout to the output of each sub-layer..."

        attn_dropout: Applied to attention weights
        - Prevents over-reliance on specific attention patterns
        - Randomly zeros some attention weights during training

        resid_dropout: Applied to output after projection
        - Part of the residual connection regularization
        """
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.resid_dropout = nn.Dropout(config.DROPOUT)

        # ===========================================================================
        # CAUSAL MASK
        # ===========================================================================

        """
        From GPT papers: Causal mask ensures autoregressive property.

        IMPLEMENTATION:
        - Lower triangular matrix: 1s on/below diagonal, 0s above
        - register_buffer: Not a learnable parameter, saved with model
        - During forward: mask[:T, :T] to handle variable length

        Why register_buffer?
        - Part of state_dict (saved with model)
        - Not updated by optimizer (not a parameter)
        - Moves to device automatically with model
        """
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
        )

    def forward(self, x):
        """
        Forward pass of causal self-attention.

        PAPER REFERENCE: "Attention is All You Need" (Section 3.2.1)

        COMPUTATION STEPS:
        ------------------
        1. Compute Q, K, V: Linear projections from input
        2. Reshape for multi-head: Split across n_head
        3. Compute attention scores: Q @ K^T / √d_k
        4. Apply causal mask: Set future positions to -∞
        5. Softmax: Convert scores to probabilities
        6. Weight values: Attention weights @ V
        7. Concatenate heads: Merge all heads
        8. Output projection: Linear transformation

        Args:
            x: Input tensor of shape (B, T, C)
               B = batch size (number of sequences)
               T = sequence length (≤ BLOCK_SIZE)
               C = embedding dimension (N_EMBD)

        Returns:
            Output tensor of shape (B, T, C) - same shape as input
            This enables residual connections: x = x + Attention(x)

        SHAPE TRANSFORMATIONS:
        ----------------------
        Input:        (B, T, C)          where C = n_embd
        Q, K, V:     (B, T, C)          after projection
        Reshaped:    (B, T, n_head, hs)  where hs = head_size = C/n_head
        Transposed:  (B, n_head, T, hs)  for efficient matmul
        Attention:   (B, n_head, T, T)   scores matrix
        After V:     (B, n_head, T, hs)  weighted values
        Transposed:  (B, T, n_head, hs)  back
        Output:      (B, T, C)           after concatenation & projection
        """
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # ===========================================================================
        # 1. COMPUTE Q, K, V
        # ===========================================================================

        """
        From "Attention is All You Need" (Section 3.2.2):
        "The queries, keys, and values... are linearly projected..."

        Single matrix multiplication produces all three:
        - Input x: (B, T, C)
        - Output: (B, T, 3×C)
        - Split into three tensors of shape (B, T, C) each

        This is equivalent to:
        Q = x @ W_Q, K = x @ W_K, V = x @ W_V
        But more efficient (single matmul + split)
        """
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # ===========================================================================
        # 2. RESHAPE FOR MULTI-HEAD
        # ===========================================================================

        """
        From "Attention is All You Need" (Section 3.2.2):
        "Instead of performing a single attention function...
        we linearly project the Q, K and V h times...

        For each projection, we use d_k = d_v = d_model/h"

        RESHAPING:
        - From (B, T, C) to (B, T, n_head, head_size)
        - Split embedding dimension across heads
        - Each head sees: head_size = C / n_head dimensions

        TRANSPOSING:
        - From (B, T, n_head, hs) to (B, n_head, T, hs)
        - Bring n_head before T for matrix operations
        - Each head processes full sequence independently
        """
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # ===========================================================================
        # 3. COMPUTE ATTENTION SCORES
        # ===========================================================================

        """
        From "Attention is All You Need" (Section 3.2.1):

        Scaled Dot-Product Attention:
            Attention(Q, K, V) = softmax(QK^T / √d_k) V

        Q @ K^T:
        - (B, n_head, T, hs) @ (B, n_head, hs, T)
        - = (B, n_head, T, T)
        - Each position t has a score for each other position

        SCALING by 1/√d_k:
        - Prevents softmax saturation when d_k is large
        - From paper: "We scale by 1/√d_k... to counteract the effect
          of having products of large dimensions"
        - When d_k is large, dot products become large
        - Large values → small gradients → poor training
        """
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)

        # ===========================================================================
        # 4. APPLY CAUSAL MASK
        # ===========================================================================

        """
        CAUSAL MASKING (from GPT papers):
        Each position can only attend to previous positions.

        Implementation:
        - mask[:T, :T]: Take T×T portion (variable length)
        - mask == 0: Find positions above diagonal (future)
        - Set to -∞: Softmax will make these zero

        VISUAL REPRESENTATION:
        For T=4, the mask looks like:
        [[1, 0, 0, 0],    ← Position 0 sees only itself
         [1, 1, 0, 0],    ← Position 1 sees positions 0, 1
         [1, 1, 1, 0],    ← Position 2 sees positions 0, 1, 2
         [1, 1, 1, 1]]    ← Position 3 sees positions 0, 1, 2, 3

        After masking, attention matrix has -∞ for future positions.
        """
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # ===========================================================================
        # 5. SOFTMAX TO GET ATTENTION WEIGHTS
        # ===========================================================================

        """
        From "Attention is All You Need":
        "Apply the softmax function to obtain the weights on the values"

        SOFTMAX:
        - Converts scores to probabilities
        - Each row sums to 1
        - -∞ becomes 0 (future positions get zero weight)

        DROPOUT:
        - From paper: "Apply dropout to the output of each sub-layer"
        - Randomly zeros some attention weights
        - Only during training (disabled in eval mode)
        """
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # ===========================================================================
        # 6. WEIGHT VALUES BY ATTENTION WEIGHTS
        # ===========================================================================

        """
        From "Attention is All You Need":
        "Multiply each value by the softmax score"

        att @ v:
        - (B, n_head, T, T) @ (B, n_head, T, hs)
        - = (B, n_head, T, hs)
        - Each position gets weighted sum of all values
        - Weights are attention probabilities from softmax

        INTUITION:
        For each position t:
        1. Look at attention weights (how much to attend to each position)
        2. Multiply values by those weights
        3. Sum up: this is the output for position t

        Example: "The cat sat"
        - Position 2 ("cat"): Might strongly attend to position 3 ("sat")
        - Position 3 ("sat"): Might attend to both "cat" and "The"
        """
        y = att @ v

        # ===========================================================================
        # 7. CONCATENATE HEADS
        # ===========================================================================

        """
        From "Attention is All You Need" (Section 3.2.2):
        "Concatenate... and project"

        TRANSPOSE: (B, n_head, T, hs) → (B, T, n_head, hs)
        VIEW: Reshape to (B, T, C) where C = n_head × hs

        This concatenates outputs from all heads.
        Each head contributes head_size dimensions.
        Total: n_head × head_size = n_embd
        """
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # ===========================================================================
        # 8. OUTPUT PROJECTION
        # ===========================================================================

        """
        From "Attention is All You Need":
        "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O"

        Final linear transformation:
        - Mixes information from all heads
        - Projects back to n_embd dimension
        - Enables information exchange across heads

        DROPOUT: Applied to output for regularization
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
    -------------------------------------------------

    From the paper:
    "In addition to attention sub-layers, each layer contains a
    fully connected feed-forward network, which is applied to
    each position separately and identically.

    FFN(x) = max(0, xW_1 + b_1) W_2 + b_2"

    ARCHITECTURE:
    ------------
    Linear(d_model, 4×d_model) → ReLU → Linear(4×d_model, d_model)

    KEY POINTS FROM PAPER:
    ----------------------
    1. "Applied to each position separately": No cross-position mixing here
       (That's what attention does!)
    2. "Identically": Same weights for all positions (parameter sharing)
    3. Two linear transformations with ReLU activation

    WHY 4× EXPANSION?
    -----------------
    From GPT-2/GPT-3: The inner dimension is 4× the embedding dimension.

    Reasons:
    1. More capacity: Larger hidden layer = more expressive power
    2. Bottleneck: Projects back to d_model (keeps parameters manageable)
    3. Empirical: Works well in practice (standard in transformers)

    For our model: n_embd = 384, inner = 4 × 384 = 1536

    RELU VS GELU:
    -------------
    Original Transformer: ReLU (Rectified Linear Unit)
    GPT-2/GPT-3: GELU (Gaussian Error Linear Unit)

    We use ReLU for simplicity, but GELU is smoother:
    ReLU: max(0, x)  (sharp corner at 0)
    GELU: x × Φ(x)  (smooth, where Φ is Gaussian CDF)

    GELU generally performs better but is slightly more expensive.
    """

    def __init__(self, config):
        super().__init__()

        # ===========================================================================
        # TWO-LAYER MLP
        # ===========================================================================

        """
        From "Attention is All You Need":
        "The dimensionality of input/output is d_model = 512,
        and the inner-layer dimensionality is d_ff = 2048"

        For GPT: d_model = n_embd, d_ff = 4 × n_embd

        LAYER 1: Expands from n_embd to 4×n_embd
        LAYER 2: Contracts from 4×n_embd back to n_embd

        This creates a "bottleneck" architecture:
        Wide middle (more capacity) → Narrow ends (same as input)
        """
        self.net = nn.Sequential(
            nn.Linear(config.N_EMBD, 4 * config.N_EMBD),
            nn.ReLU(),  # Non-linearity is crucial! Without it: just linear model
            nn.Linear(4 * config.N_EMBD, config.N_EMBD),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# PART 3: TRANSFORMER BLOCK
# =============================================================================


class TransformerBlock(nn.Module):
    """
    Complete Transformer Block

    PAPER: "Attention is All You Need" (Section 3.1)
    -----------------------------------------------

    From the paper, the Transformer block consists of:
    1. Multi-head attention
    2. Add & Norm (residual + layer normalization)
    3. Feed-forward network
    4. Add & Norm (residual + layer normalization)

    ARCHITECTURE VARIANTS:
    ---------------------
    Original Transformer (2017):
        x = LayerNorm(x + MultiHeadAttention(x))
        x = LayerNorm(x + FFN(x))
        → Post-LN: LayerNorm AFTER sublayer

    GPT-2 (2019):
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
        → Pre-LN: LayerNorm BEFORE sublayer

    We use GPT-2 style (Pre-LN) because:
    1. More stable for deep networks
    2. Better gradient flow
    3. Standard in modern language models

    RESIDUAL CONNECTIONS:
    --------------------
    From "Attention is All You Need":
    "We employ residual connections around each of the two sub-layers"

    RESIDUAL: x = x + Sublayer(x)

    Why residuals?
    1. Gradient flow: Create "highways" for gradients to flow backward
    2. Training stability: Allow network to learn identity function easily
    3. Depth: Enable training of very deep networks (100+ layers)

    Without residuals, deep networks suffer from:
    - Vanishing gradients: Gradients become tiny, learning stops
    - Exploding gradients: Gradients become huge, training diverges

    LAYER NORMALIZATION:
    -------------------
    From "Layer Normalization" (Ba et al., 2016)

    Normalizes across the feature dimension:
        y = γ × (x - μ) / σ + β

    Where:
    - μ: Mean of features
    - σ: Standard deviation of features
    - γ: Learnable scale (gain)
    - β: Learnable shift (bias)

    Why LayerNorm (not BatchNorm)?
    1. Batch size independence: Works well with small batches
    2. Sequential data: Designed for RNNs/transformers
    3. Stability: More stable for variable-length sequences
    """

    def __init__(self, config):
        super().__init__()

        # ===========================================================================
        # ATTENTION AND FFN
        # ===========================================================================

        """
        The two core sublayers:
        1. Self-attention: Aggregates information across positions
        2. Feed-forward: Processes information at each position
        """
        self.attn = CausalSelfAttention(config)
        self.ffwd = FeedForward(config)

        # ===========================================================================
        # LAYER NORMALIZATION
        # ===========================================================================

        """
        Two LayerNorm layers:
        - ln1: Before attention
        - ln2: Before FFN

        From GPT-2: Each sublayer has its own LayerNorm
        """
        self.ln1 = nn.LayerNorm(config.N_EMBD)
        self.ln2 = nn.LayerNorm(config.N_EMBD)

    def forward(self, x):
        """
        Forward pass with residual connections.

        PRE-LN ARCHITECTURE (GPT-2 style):
        -----------------------------------
        x = x + Attention(LayerNorm(x))  # Attention with residual
        x = x + FFN(LayerNorm(x))        # FFN with residual

        ORDER OF OPERATIONS:
        1. LayerNorm: Normalize features
        2. Sublayer: Apply attention or FFN
        3. Add: Residual connection

        This is different from Post-LN:
        Post-LN: x = LayerNorm(x + Sublayer(x))
        Pre-LN:  x = x + Sublayer(LayerNorm(x))

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C) - same shape as input
        """
        # ===========================================================================
        # SELF-ATTENTION WITH RESIDUAL
        # ===========================================================================

        """
        x = x + Attention(LayerNorm(x))

        STEP 1: LayerNorm(x)
        - Normalizes features to zero mean, unit variance
        - Stabilizes training

        STEP 2: Attention(LayerNorm(x))
        - Applies self-attention
        - Aggregates information across sequence

        STEP 3: x + Attention(...)
        - Residual connection
        - Preserves original information
        - Enables gradient flow
        """
        x = x + self.attn(self.ln1(x))

        # ===========================================================================
        # FEED-FORWARD WITH RESIDUAL
        # ===========================================================================

        """
        x = x + FFN(LayerNorm(x))

        Same pattern as attention:
        1. Normalize
        2. Apply FFN
        3. Add residual
        """
        x = x + self.ffwd(self.ln2(x))

        return x


# =============================================================================
# PART 4: COMPLETE GPT MODEL
# =============================================================================


class GPT(nn.Module):
    """
    Generative Pre-trained Transformer (GPT) Model

    PAPERS:
    -------
    1. "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
       Radford et al., OpenAI

    2. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
       Radford et al., OpenAI

    ARCHITECTURE OVERVIEW:
    --------------------
    From GPT-1:
    "We use a 12-layer decoder-only transformer...
    with masked self-attention heads (12 heads for the largest model)..."

    Full architecture:
    1. Token Embedding: Convert indices to vectors
    2. Positional Embedding: Add position information
    3. Transformer Blocks: Stack of N blocks
    4. Final LayerNorm: Stabilize output
    5. Language Model Head: Project to vocabulary

    EMBEDDINGS:
    ----------
    TOKEN EMBEDDING:
    From GPT-1: "We use learned position embeddings... and learned
    token embeddings"

    Each token (character) gets a learnable vector of size n_embd.
    This is a lookup table: vocab_size × n_embd matrix.

    POSITIONAL EMBEDDING:
    From "Attention is All You Need":
    Original: Sinusoidal position encodings
    GPT: Learned position embeddings

    Why learned?
    1. Simpler: Easier to implement and optimize
    2. Flexible: Can learn position-specific patterns
    3. Empirical: Works as well as sinusoidal

    ADDING (NOT CONCATENATING):
    token_embedding + positional_embedding

    From "Attention is All You Need":
    "We use the same learned weight matrix for both"

    Why add?
    - Each position sees "what" (token) + "where" (position)
    - Same token at different position = different vector
    - Keeps dimensionality manageable (vs concatenation)

    GENERATION (INFERENCE):
    ---------------------
    From GPT-2: "We use a top-k of k=40... for sampling"

    Autoregressive generation:
    1. Feed context: Sequence of existing tokens
    2. Forward pass: Get predictions for next token
    3. Sample: Pick token from probability distribution
    4. Append: Add sampled token to sequence
    5. Repeat: Continue until desired length

    This is greedy/iterative decoding - one token at a time.
    (Beam search is another option but more expensive)
    """

    def __init__(self, config):
        super().__init__()

        # Store config for saving/loading
        self.config = config

        # ===========================================================================
        # EMBEDDINGS
        # ===========================================================================

        """
        From GPT-1: "We use learned position embeddings... and learned
        token embeddings... The sum of these embeddings is the input
        to the transformer blocks."

        IMPLEMENTATION:
        - wte: Weight TEmbedding (token embeddings)
        - wpe: Weight PEmbedding (positional embeddings)
        - drop: Dropout on embeddings (regularization)
        """
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.VOCAB_SIZE, config.N_EMBD),  # Token embeddings
                wpe=nn.Embedding(
                    config.BLOCK_SIZE, config.N_EMBD
                ),  # Positional embeddings
                drop=nn.Dropout(config.DROPOUT),
            )
        )

        # ===========================================================================
        # TRANSFORMER BLOCKS
        # ===========================================================================

        """
        From GPT-1:
        "The model uses a 12-layer transformer... For the largest model,
        we use 12 attention heads"

        From GPT-2:
        Model sizes: 117M (12 layers), 345M (24 layers), 774M (36 layers), 1.5B (48 layers)

        IMPLEMENTATION:
        - Stack of N transformer blocks
        - Each block: attention + FFN with residuals
        - Sequential processing (not parallel)
        """
        blocks = [TransformerBlock(config) for _ in range(config.N_LAYER)]
        self.transformer.blocks = nn.ModuleList(blocks)

        # ===========================================================================
        # OUTPUT HEAD
        # ===========================================================================

        """
        From GPT-1:
        "The output of the transformer stack is a hidden vector of
        dimension d_model... We project this... to vocabulary size"

        TWO COMPONENTS:
        1. Final LayerNorm: Stabilize before output
        2. Language Model Head: Project to vocabulary logits

        LM HEAD:
        Linear transformation: d_model → vocab_size
        Produces logits (unnormalized scores) for each token in vocabulary
        """
        self.transformer.ln_f = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, config.VOCAB_SIZE, bias=False)

        # ===========================================================================
        # WEIGHT TYING (OPTIONAL)
        # ===========================================================================

        """
        From "Using the Output Embedding to Improve Language Models"
        (Press & Wolf, 2017)

        Weight tying: Share embedding and output matrices

        Standard: W_emb ≠ W_out
        Weight tying: W_emb = W_out

        Benefits:
        1. Fewer parameters: Save vocab_size × n_embd parameters
        2. Regularization: Forces representation consistency
        3. Empirical: Often improves performance

        We don't use it for character-level (vocab is small anyway).
        Uncomment for large vocabularies (like BPE):
        # self.lm_head.weight = self.transformer.wte.weight
        """

        # ===========================================================================
        # WEIGHT INITIALIZATION
        # ===========================================================================

        """
        From GPT-2: "We use the same initialization as in GPT-2"

        Proper initialization is CRUCIAL for deep networks!
        Bad init → vanishing/exploding gradients → training fails

        From "Delving Deep into Rectifiers" (He et al., 2015):
        Normal(0, √2/fan_in) for ReLU networks

        GPT-2 uses: Normal(0, 0.02) for all linear layers
        """
        self.apply(self._init_weights)

        print(f"Model initialized with {self.get_num_params():,} parameters")

    def _init_weights(self, module):
        """
        Initialize weights following GPT-2 conventions.

        PAPER REFERENCE: GPT-2 (2019)

        From GPT-2 paper and code:
        "We use a simplified version of the initialization from GPT-2"

        INITIALIZATION SCHEME:
        ---------------------
        1. Linear layers: Normal(0, 0.02)
        2. Embeddings: Normal(0, 0.02)
        3. LayerNorm: γ=1, β=0 (identity transform)

        WHY NORMAL(0, 0.02)?
        -------------------
        - Small random values prevent saturation
        - Prevents exploding gradients at start
        - 0.02 is a heuristic that works well

        WHY LAYERNORM(γ=1, β=0)?
        ----------------------------
        - Identity transformation initially
        - LayerNorm passes through unchanged at start
        - Network can learn to deviate from identity

        Args:
            module: Neural network module to initialize
        """
        if isinstance(module, nn.Linear):
            # He initialization variant
            # From "Delving Deep into Rectifiers"
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Small random initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Identity transform initially
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        From GPT-2 paper: Parameter counts for different model sizes

        Args:
            non_embedding: If False, count embedding params too

        Note: We usually exclude token embeddings from parameter count
        because they're a lookup table, not "learned" parameters in the
        traditional sense (though they do update during training).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        PAPER: "Language Models are Unsupervised Multitask Learners"

        FORWARD PASS:
        1. Lookup token embeddings
        2. Add positional embeddings
        3. Pass through transformer blocks
        4. Final layer norm
        5. Project to vocabulary (logits)
        6. Compute loss (if targets provided)

        Args:
            idx: Input token indices of shape (B, T)
                 B = batch size
                 T = sequence length (≤ BLOCK_SIZE)
            targets: Target token indices of shape (B, T)
                     Only needed during training for loss computation

        Returns:
            logits: Output logits of shape (B, T, vocab_size)
                    Logits for each position, for each token in vocabulary
            loss: Cross-entropy loss (scalar, None if targets not provided)

        SHAPE FLOW:
        -----------
        Input:  (B, T)              - token indices
        Emb:    (B, T, C)            - token + pos embeddings
        Blocks: (B, T, C)            - after transformer blocks
        Logits: (B, T, vocab_size)   - vocabulary projections
        Loss:   scalar              - cross-entropy (if training)
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
        From GPT-1:
        "The first layer is a token embedding matrix"

        Lookup: Each token index gets its corresponding embedding vector
        (B, T) → (B, T, n_embd)

        This is a learned lookup table.
        Initially random, learned during training.
        """
        tok_emb = self.transformer.wte(idx)

        # ===========================================================================
        # 2. POSITIONAL EMBEDDINGS
        # ===========================================================================

        """
        From GPT-1:
        "The second layer is a position embedding matrix"

        Create position indices: [0, 1, 2, ..., T-1]
        Look up positional embeddings
        (T,) → (T, n_embd) → broadcasts to (B, T, n_embd)
        """
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        # ===========================================================================
        # 3. COMBINE EMBEDDINGS
        # ===========================================================================

        """
        From GPT-1:
        "The sum of these embeddings is the input to the transformer"

        x = token_embedding + positional_embedding

        Why add not concatenate?
        - Keeps dimensionality fixed (n_embd)
        - Enables generalization across positions
        - Same token at different position = different vector

        Example: "cat" at position 0 ≠ "cat" at position 5
        """
        x = self.transformer.drop(tok_emb + pos_emb)

        # ===========================================================================
        # 4. TRANSFORMER BLOCKS
        # ===========================================================================

        """
        From GPT-1:
        "The transformer stacks... masked self-attention layers"

        Pass through each block sequentially:
        - Each block: attention + FFN with residuals
        - Output of block i = input to block i+1
        - Information flows and mixes through depth
        """
        for block in self.transformer.blocks:
            x = block(x)

        # ===========================================================================
        # 5. FINAL LAYER NORM
        # ===========================================================================

        """
        From "Attention is All You Need":
        "At the end... we apply layer normalization"

        Normalizes features before final projection.
        Stabilizes training and improves generalization.
        """
        x = self.transformer.ln_f(x)

        # ===========================================================================
        # 6. PROJECT TO VOCABULARY (LOGITS)
        # ===========================================================================

        """
        From GPT-1:
        "We project the hidden state to vocabulary size"

        Linear transformation: n_embd → vocab_size
        (B, T, n_embd) → (B, T, vocab_size)

        Logits: Unnormalized scores
        High logit = high probability (after softmax)
        """
        logits = self.lm_head(x)

        # ===========================================================================
        # 7. LOSS COMPUTATION (training only)
        # ===========================================================================

        """
        From GPT-1:
        "We optimize the language modeling objective"

        Cross-entropy loss:
        L = -Σ y_true * log(softmax(logits))

        For language modeling:
        - y_true: One-hot encoding of actual next token
        - y_pred: Softmax of predicted logits
        - Loss: Negative log-likelihood of correct token

        Reshaping for PyTorch:
        - PyTorch expects (N, C) for logits and (N,) for targets
        - We reshape: (B, T, vocab_size) → (B×T, vocab_size)
        - Targets: (B, T) → (B×T,)

        ignore_index=-1:
        Don't compute loss for padding tokens (if any)
        """
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        PAPER: "Language Models are Unsupervised Multitask Learners"

        From GPT-2:
        "For sampling... we use nucleus sampling... with p=0.9"
        "We also use top-k... with k=40"

        AUTOREGRESSIVE GENERATION:
        -------------------------
        To generate token t+1:
        1. Use tokens [0, 1, ..., t] as context
        2. Forward pass to get distribution for token t+1
        3. Sample from distribution
        4. Append sampled token to context
        5. Repeat

        This is greedy decoding - one token at a time.
        Alternatives: Beam search, nucleus sampling, etc.

        TEMPERATURE:
        Controls randomness of sampling
        - < 1.0: More conservative, peakier distribution
        - = 1.0: Standard sampling from predicted distribution
        - > 1.0: More diverse, flatter distribution

        Formula: logits / temperature
        Lower temp → larger logits → more extreme softmax
        Higher temp → smaller logits → more uniform softmax

        TOP-K SAMPLING:
        From "Curious Sampling" (Fan et al., 2018)

        Only sample from top k most likely tokens.
        - Reduces probability of sampling nonsense
        - Balances quality and diversity
        - k=0 means disable (sample from all tokens)

        Args:
            idx: Input context of shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = disable)

        Returns:
            Generated sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # =======================================================================
            # CROP CONTEXT (for efficiency)
            # =======================================================================

            """
            If sequence is too long, crop to block_size.
            We only need the last block_size tokens for next prediction
            (due to causal masking - earlier tokens don't affect last token)
            """
            idx_crop = (
                idx
                if idx.size(1) <= self.config.BLOCK_SIZE
                else idx[:, -self.config.BLOCK_SIZE :]
            )

            # =======================================================================
            # FORWARD PASS
            # =======================================================================

            """
            Get predictions for next token.
            We only use the last position's output (next token prediction).
            """
            logits, _ = self(idx_crop)

            # =======================================================================
            # GET LOGITS FOR LAST POSITION
            # =======================================================================

            """
            logits[:, -1, :]: Logits for next token only

            Shape: (B, vocab_size)
            - B: Different predictions for each sequence in batch
            - vocab_size: Score for each token in vocabulary
            """
            logits = logits[:, -1, :] / temperature

            # =======================================================================
            # TOP-K SAMPLING (optional)
            # =======================================================================

            """
            If top_k is set, only sample from top k tokens.

            1. Get top k logits and their indices
            2. Set all other logits to -∞
            3. After softmax, these have zero probability

            This prevents sampling from very low-probability tokens
            that might be nonsensical.
            """
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # =======================================================================
            # SAMPLE FROM DISTRIBUTION
            # =======================================================================

            """
            Convert logits to probabilities: softmax
            Sample from categorical distribution

            multinomial: Sample from multinomial distribution
            Equivalent to: Pick token according to its probability
            """
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # =======================================================================
            # APPEND TO SEQUENCE
            # =======================================================================

            """
            Append sampled token to sequence.
            This extended sequence becomes the new context.
            """
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
