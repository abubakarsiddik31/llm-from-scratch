"""
Training Script for Character-Level GPT

This script implements the training pipeline following the GPT architecture
from "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
and "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019).

PAPER REFERENCE: "Language Models are Unsupervised Multitask Learners"
Authors: Radford et al. (OpenAI, 2019)
Key ideas we implement:
- Autoregressive language modeling (next-token prediction)
- Transformer decoder architecture (Attention is All You Need, 2017)
- AdamW optimizer with weight decay
- Learning rate warmup (we use constant LR for simplicity)
- Batch size and sequence length (context window)

TRAINING PIPELINE:
1. Data loading and preprocessing (character-level tokenization)
2. Train/validation split (90/10) for overfitting detection
3. Batch creation with causal targets (shifted by 1)
4. Training loop: forward → backward → optimizer step
5. Validation and checkpointing
6. Model saving for inference
"""

import os
import torch
from torch.nn import functional as F

import config
from model import GPT


# =============================================================================
# PART 1: DATA LOADING
# =============================================================================

def load_data(data_path):
    """
    Load text data and create train/val splits.

    PAPER CONTEXT: Language Modeling Objective
    ------------------------------------------
    From GPT-1 & GPT-2 papers: The model learns by maximizing the likelihood
    of sequential text. For a sequence of tokens (x1, x2, ..., xn):

        L = -Σ log P(xi | x1, x2, ..., xi-1)

    This is the "autoregressive" objective - predict each token given
    all previous tokens. The training data provides the "ground truth"
    for each position.

    WHY TRAIN/VAL SPLIT?
    -------------------
    From "Improving Language Understanding..." (GPT-1, 2018):
    Models are evaluated on held-out data to measure generalization.

    Split: 90% train, 10% validation
    - If train_loss >> val_loss: Underfitting (model can't learn)
    - If train_loss << val_loss: Overfitting (memorizing)

    CHARACTER-LEVEL TOKENIZATION:
    ----------------------------
    GPT-1/GPT-2 use Byte-Pair Encoding (BPE) with vocab size ~50k.
    We use character-level (vocab ~65) for:
    1. Simplicity - no need for separate tokenizer training
    2. Faster training - smaller vocabulary means smaller output layer
    3. Educational - easier to inspect what model learns

    Trade-off: Character-level needs deeper/wider models for same performance.
    """
    print(f"Loading data from {data_path}...")

    # Load raw text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")

    # ============================================================================
    # VOCABULARY CREATION
    # ============================================================================

    # In GPT papers, vocabulary is learned via BPE from training data.
    # Here, our vocabulary is simply all unique characters.

    chars = sorted(list(set(text)))  # Unique characters = vocabulary
    vocab_size = len(chars)
    config.VOCAB_SIZE = vocab_size

    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {''.join(chars)}")

    # ============================================================================
    # ENCODING/DECODING MAPPINGS
    # ============================================================================

    # stoi: string to index (encoding)
    # itos: index to string (decoding)
    #
    # In GPT-2, this is handled by the BPE tokenizer (tokenizers/byte-level-bpe)
    # We use a simple dictionary for character-level.

    stoi = {ch: i for i, ch in enumerate(chars)}  # {'\n': 0, ' ': 1, '!': 2, ...}
    itos = {i: ch for i, ch in enumerate(chars)}  # {0: '\n', 1: ' ', 2: '!', ...}

    # Encode: convert string "Hello" → [H, e, l, l, o] → [30, 40, 50, 50, 55]
    # This is what GPT-2's tokenizer.encode() does
    encode = lambda s: [stoi[c] for c in s]

    # Decode: convert [30, 40, 50, 50, 55] → "Hello"
    # This is what GPT-2's tokenizer.decode() does
    decode = lambda l: ''.join([itos[i] for i in l])

    # ============================================================================
    # DATA ENCODING
    # ============================================================================

    # Convert entire text to integer tensor
    # This is our "dataset" - a single long sequence of token IDs
    data = torch.tensor(encode(text), dtype=torch.long)

    # ============================================================================
    # TRAIN/VAL SPLIT
    # ============================================================================

    # GPT papers use held-out sets for evaluation:
    # - GPT-1: BookCorpus for training, various test sets for eval
    # - GPT-2: WebText (8M documents) with random document-level split
    #
    # We use a simple 90/10 character-level split (not document-level,
    # since Shakespeare is one continuous text)

    n = int(0.9 * len(data))
    train_data = data[:n]   # First 90% for training
    val_data = data[n:]     # Last 10% for validation

    print(f"Train data: {len(train_data):,} characters")
    print(f"Val data:   {len(val_data):,} characters")

    return train_data, val_data, encode, decode


# =============================================================================
# PART 2: BATCH CREATION (DATA LOADING)
# =============================================================================

def get_batch(split):
    """
    Create a training batch with causal (shifted) targets.

    PAPER CONTEXT: Causal Language Modeling
    --------------------------------------
    From "Attention is All You Need" (2017):
    In the decoder, positions can attend to all previous positions.
    This creates the "causal" (autoregressive) property.

    The training objective is:
        P(x_t | x_1, x_2, ..., x_{t-1})

    For a sequence "Hello", we create:
        Input:  [H, e, l, l, o]
        Target: [e, l, l, o, <next>]

    At each position t:
    - Input: tokens from 1 to t
    - Target: token at position t+1
    - Model learns: predict next token given previous tokens

    IMPLEMENTATION:
    --------------
    Given a long sequence of tokens (our encoded text):
    1. Randomly sample BATCH_SIZE starting positions
    2. Extract BLOCK_SIZE consecutive tokens as input (x)
    3. Extract the next BLOCK_SIZE tokens as targets (y)
    4. Targets are inputs shifted by 1 position!

    This is called "teacher forcing" - we provide the ground truth
    for each position during training.

    WHY RANDOM STARTING POSITIONS?
    ----------------------------
    From GPT-2: "We use a byte-level version of Byte-Pair Encoding (BPE)...
    and document-level shuffling..."

    Random sampling ensures:
    1. Each batch sees different parts of the dataset
    2. Model learns from all positions (not just beginning of sequences)
    3. Better GPU utilization (can shuffle arbitrarily)

    BATCH SIZE & CONTEXT LENGTH:
    ---------------------------
    BATCH_SIZE = 64: Number of sequences processed in parallel
    BLOCK_SIZE = 256: Maximum context length (sequence length)

    From GPT-2:
    - Context window: 1024 tokens
    - Batch size: 512 sequences × 1024 tokens = 524K tokens/batch

    We use smaller values for faster training on consumer hardware.

    Args:
        split: 'train' or 'val' - which dataset to sample from

    Returns:
        x: Input tensor (B, T) where B=BATCH_SIZE, T=BLOCK_SIZE
        y: Target tensor (B, T) - same shape, shifted by 1
    """
    data = train_data if split == 'train' else val_data

    # ===========================================================================
    # RANDOM STARTING POSITION SAMPLING
    # ===========================================================================

    # Sample random starting indices for each sequence in the batch
    # We can't start past len(data) - BLOCK_SIZE (need BLOCK_SIZE tokens)
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))

    # ===========================================================================
    # EXTRACT INPUT SEQUENCES (x)
    # ===========================================================================

    # For each starting position, extract BLOCK_SIZE consecutive tokens
    # Stack creates batch dimension
    x = torch.stack([data[i:i + config.BLOCK_SIZE] for i in ix])
    # Shape: (BATCH_SIZE, BLOCK_SIZE) = (64, 256)

    # ===========================================================================
    # EXTRACT TARGET SEQUENCES (y) - SHIFTED BY 1
    # ===========================================================================

    # Targets are the NEXT tokens at each position
    # If x = [x1, x2, x3, ..., xT], then y = [x2, x3, x4, ..., x{T+1}]
    y = torch.stack([data[i + 1:i + config.BLOCK_SIZE + 1] for i in ix])
    # Shape: (BATCH_SIZE, BLOCK_SIZE) = (64, 256)

    # Example with BLOCK_SIZE=5:
    # x: "Hello"  → [H, e, l, l, o]
    # y: "ello_"  → [e, l, l, o, <next>]
    #
    # At position 0: input is 'H', target is 'e' (predict 'e' given 'H')
    # At position 1: input is 'He', target is 'l' (predict 'l' given 'He')
    # At position 4: input is 'Hello', target is <next> (predict next token)

    # ===========================================================================
    # MOVE TO DEVICE (GPU)
    # ===========================================================================

    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    return x, y


# =============================================================================
# PART 3: LOSS ESTIMATION (VALIDATION)
# =============================================================================

@torch.no_grad()
def estimate_loss():
    """
    Estimate average loss on train and validation sets.

    PAPER CONTEXT: Evaluation Metrics
    --------------------------------
    From GPT-2 paper:
    "We evaluate using zero-shot performance on downstream tasks..."

    For language modeling, the standard metric is **perplexity**:
        Perplexity = exp(cross_entropy_loss)

    Lower loss → lower perplexity → better model.
    Human-level perplexity on English is ~10-20 (loss ~2.3-3.0).

    WHY ESTIMATE LOSS (NOT SINGLE BATCH)?
    ------------------------------------
    Single batch loss is noisy (varies based on which sequences sampled).
    We average over EVAL_ITERS batches for stable estimate.

    From GPT papers: Validation loss is computed on held-out set,
    typically the full validation set. We use random sampling.

    WHY @torch.no_grad?
    ------------------
    Disables gradient computation for this function:
    1. No backward pass needed (just evaluation)
    2. Saves memory (don't store intermediate activations)
    3. Faster computation (skip gradient bookkeeping)

    MODEL.EVAL() VS MODEL.TRAIN():
    -----------------------------
    - model.eval(): Disables dropout (deterministic predictions)
    - model.train(): Enables dropout (regularization during training)

    Important: We switch back to train mode after evaluation!
    """
    model.eval()  # Disable dropout for consistent evaluation
    losses = {}

    for split in ['train', 'val']:
        # Average loss over multiple batches
        losses_split = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses_split[k] = loss.item()
        losses[split] = losses_split.mean()

    model.train()  # Re-enable dropout for training
    return losses


# =============================================================================
# PART 4: MAIN TRAINING LOOP
# =============================================================================

def train(model, train_data, val_data, decode, checkpoint_dir):
    """
    Main training loop implementing the GPT training procedure.

    PAPER CONTEXT: Training Procedure
    --------------------------------
    From "Improving Language Understanding..." (GPT-1, 2018):
    "We train a 12-layer decoder-only transformer..."

    From GPT-2 (2019):
    "We trained a 1.5B parameter model... on WebText..."

    Key training components from papers:
    1. Optimizer: AdamW (Adam + decoupled weight decay)
    2. Learning rate: 2.5e-4 (GPT-2) with warmup
    3. Batch size: 512 sequences × 1024 context (GPT-2)
    4. Gradient clipping: 1.0 (prevent exploding gradients)
    5. Learning rate schedule: Cosine decay (GPT-3)

    Our implementation:
    - AdamW optimizer (matches GPT-2)
    - Constant LR 3e-4 (simplified, no warmup/decay for this tutorial)
    - Small batch size (64 × 256) due to GPU constraints
    - No gradient clipping (not needed for this scale)

    THE TRAINING LOOP:
    -----------------
    Standard gradient descent loop:
        1. Sample batch: Get (x, y) pairs
        2. Forward pass: Compute predictions and loss
        3. Backward pass: Compute gradients (dLoss/dParams)
        4. Optimizer step: Update params using gradients

    This is repeated for MAX_ITERS iterations.

    CROSS-ENTROPY LOSS:
    -----------------
    Loss = -Σ y_true * log(y_pred)

    For language modeling:
    - y_true: One-hot encoding of actual next token
    - y_pred: Predicted probability distribution (softmax output)
    - Loss: Negative log-likelihood of correct token

    Minimizing this = Maximizing likelihood of correct predictions.
    """
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # ===========================================================================
    # OPTIMIZER SETUP
    # ===========================================================================

    """
    PAPER: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

    Adam (Adaptive Moment Estimation) combines:
    1. Momentum: Moving average of gradients (accelerates convergence)
    2. RMSprop: Moving average of squared gradients (adapts learning rate)

    AdamW (Decoupled Weight Decay):
    Paper: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)

    Key difference from Adam:
    - Adam: L2 regularization applied to gradients
    - AdamW: L2 regularization applied to parameters directly

    From GPT-2: "We use Adam with β1=0.9, β2=0.999"

    HYPERPARAMETERS:
    ---------------
    - lr=3e-4: Learning rate (step size for parameter updates)
               GPT-2 uses 2.5e-4, we use 3e-4 (common default)
    - betas=(0.9, 0.999): Adam momentum parameters
      - β1=0.9: Exponential decay rate for first moment (gradient mean)
      - β2=0.999: Exponential decay rate for second moment (gradient variance)
    - weight_decay=0.1: L2 regularization strength
      - Penalizes large weights (prevents overfitting)
      - GPT-2 uses 0.01 for some models, 0.1 for others

    RELATION TO PAPER:
    -----------------
    From GPT-2: "All models use the same optimizer hyperparameters:
    β1 = 0.9, β2 = 0.999, ε = 1e-8, and weight decay of 0.01"
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,      # 3e-4
        betas=(0.9, 0.999),           # Standard Adam values
        weight_decay=0.1              # L2 regularization
    )

    # ===========================================================================
    # TRAINING STATE
    # ===========================================================================

    best_val_loss = float('inf')  # Track best validation loss for checkpointing

    # ===========================================================================
    # MAIN TRAINING LOOP
    # ===========================================================================

    for iter in range(config.MAX_ITERS):

        # =======================================================================
        # EVALUATION (PERIODIC)
        # =======================================================================

        """
        PAPER CONTEXT: Checkpointing and Evaluation
        --------------------------------------------
        From GPT papers: Models are evaluated periodically during training.
        GPT-2 saved checkpoints at regular intervals.

        Why evaluate periodically?
        1. Track overfitting (train_loss vs val_loss gap)
        2. Early stopping (stop if validation loss stops improving)
        3. Select best model (lowest validation loss)
        4. Monitor training progress

        EVAL_INTERVAL = 500: Evaluate every 500 iterations
        """
        if iter % config.EVAL_INTERVAL == 0 or iter == config.MAX_ITERS - 1:
            losses = estimate_loss()

            print(f"\nStep {iter}:")
            print(f"  Train loss: {losses['train']:.4f}")
            print(f"  Val loss:   {losses['val']:.4f}")

            # ===================================================================
            # GENERATE SAMPLE TEXT
            # ===================================================================

            """
            QUALITATIVE EVALUATION:
            ----------------------
            From GPT-2: "We show text generated from our models..."

            Generating sample text shows:
            1. Coherence: Does text make sense?
            2. Structure: Punctuation, dialogue, formatting
            3. Style: Shakespeare-like language?

            Temperature sampling (in model.generate()):
            - Lower temperature: More conservative, repetitive
            - Higher temperature: More diverse, creative
            - We use default temperature=1.0 here
            """
            if iter > 0:
                model.eval()
                # Start from empty context (generate from scratch)
                context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
                generated = model.generate(context, max_new_tokens=100)[0].tolist()
                print(f"  Sample:    {decode(generated)}")
                model.train()

            # ===================================================================
            # CHECKPOINTING
            # ===================================================================

            """
            Save model if validation loss improves.
            This gives us the "best" model (lowest val loss).
            """
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter{iter}.pt')
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                }, checkpoint_path)
                print(f"  → Saved checkpoint: {checkpoint_path}")

        # =======================================================================
        # TRAINING STEP
        # =======================================================================

        """
        ONE TRAINING ITERATION:
        ----------------------
        Implements standard gradient descent:

        Paper: "Attention is All You Need" (2017)
        "We use the Adam optimizer with β1=0.9, β2=0.999..."

        Steps:
        1. Get batch: Sample (x, y) pairs
        2. Forward pass: Compute predictions and loss
        3. Zero gradients: Clear previous iteration's gradients
        4. Backward pass: Compute gradients (dLoss/dParams)
        5. Optimizer step: Update parameters using gradients

        GRADIENT COMPUTATION:
        --------------------
        loss.backward() computes:
        ∂L/∂w for all parameters w using backpropagation.

        For transformer, this involves:
        - Loss → Softmax → Logits
        - Logits → LayerNorm → FFN
        - FFN → LayerNorm → Attention
        - Attention → LayerNorm → Embeddings

        The chain rule is applied through all layers.
        """

        # 1. Get batch
        xb, yb = get_batch('train')

        # 2. Forward pass
        # model(xb, yb) computes:
        # - Embedding lookup (token + positional)
        # - Transformer blocks (attention + FFN)
        # - Output projection to vocabulary
        # - Cross-entropy loss
        logits, loss = model(xb, yb)

        # 3. Clear previous gradients
        """
        set_to_none=True: More efficient than setting to zero
        - Avoids writing to gradient memory
        - Memory allocator can reuse the memory

        PAPER: PyTorch uses this by default in modern versions
        """
        optimizer.zero_grad(set_to_none=True)

        # 4. Backward pass
        """
        Computes gradients via backpropagation:
        - For each parameter w: grad_w = ∂L/∂w
        - Uses automatic differentiation (autograd)
        - Stores gradients in parameter.grad attribute

        From GPT papers: No gradient clipping at this scale
        (GPT-2 uses gradient clipping of 1.0 for large models)
        """
        loss.backward()

        # 5. Update parameters
        """
        AdamW update rule (simplified):
        1. Compute biased first moment estimate: m = β1*m + (1-β1)*g
        2. Compute biased second moment estimate: v = β2*v + (1-β2)*g²
        3. Bias-corrected estimates: m̂ = m/(1-β1^t), v̂ = v/(1-β2^t)
        4. Parameter update: w = w - lr * m̂ / (√v̂ + ε) - lr*λ*w

        Where:
        - g: Current gradient
        - m: Running mean of gradients
        - v: Running variance of gradients
        - lr: Learning rate
        - λ: Weight decay
        - t: Timestep (iteration)
        """
        optimizer.step()

        # =======================================================================
        # PROGRESS TRACKING
        # =======================================================================

        # Print loss every 100 iterations
        # This shows training progress in real-time
        if iter % 100 == 0:
            print(f"Iter {iter}/{config.MAX_ITERS} | Loss: {loss.item():.4f}")

    # ===========================================================================
    # FINAL SAVE
    # ===========================================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    final_path = os.path.join(checkpoint_dir, 'model_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, final_path)
    print(f"Saved final model to {final_path}")


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # ===========================================================================
    # CONFIGURATION VALIDATION
    # ===========================================================================

    """
    Validate hyperparameters are consistent.
    For example: n_embd must be divisible by n_head.
    """
    config.validate_config()
    config.print_config()

    # ===========================================================================
    # SETUP DIRECTORIES
    # ===========================================================================

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'checkpoints',
        'project1'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ===========================================================================
    # LOAD DATA
    # ===========================================================================

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data',
        'shakespeare.txt'
    )

    if not os.path.exists(data_path):
        print(f"\nData file not found at {data_path}")
        print("Please download Shakespeare text:")
        print("  wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare.txt")
        print("Or run: python phase1_foundation/project1_minimal_gpt/download_data.py")
        exit(1)

    global train_data, val_data
    train_data, val_data, encode, decode = load_data(data_path)

    # ===========================================================================
    # CREATE MODEL
    # ===========================================================================

    """
    MODEL INITIALIZATION:
    --------------------
    From GPT papers: "We initialize weights as in GPT-2..."

    Weight initialization is crucial for training stability:
    - Too large: Gradients explode, training diverges
    - Too small: Gradients vanish, slow convergence

    Our _init_weights() function (in model.py) uses:
    - Linear layers: Normal(0, 0.02)
    - Embeddings: Normal(0, 0.02)
    - LayerNorm: γ=1, β=0 (identity transform)

    These are from GPT-2's initialization scheme.
    """
    print("\nCreating model...")
    model = GPT(config)
    model.to(config.DEVICE)
    model.train()  # Set to training mode (enables dropout)

    # ===========================================================================
    # PARAMETER COUNT
    # ===========================================================================

    """
    From GPT-2: Model sizes from 117M to 1.5B parameters.

    Our model: ~10M parameters (tiny in comparison!)

    Parameter breakdown:
    - Embeddings: vocab_size × n_embd ≈ 65 × 384 = 25K
    - Positional: block_size × n_embd = 256 × 384 = 98K
    - Attention (per layer): 4 × n_embd² = 4 × 384² = 590K
    - FFN (per layer): 8 × n_embd² = 8 × 384² = 1.2M
    - Total per layer: ~1.8M
    - 6 layers: ~10.8M parameters
    """
    num_params = model.get_num_params()
    print(f"Total parameters: {num_params:,}")

    # ===========================================================================
    # TRAIN
    # ===========================================================================

    """
    PAPER: "Improving Language Understanding..." (GPT-1, 2018)

    "We train the model for 10 epochs on BookCorpus..."

    For our case:
    - MAX_ITERS = 5000 iterations
    - Each iteration: 64 sequences × 256 tokens = 16,384 tokens
    - Total tokens seen: 5000 × 16,384 = 81.9M tokens
    - Shakespeare: ~1M characters
    - So we see the dataset ~80 times (equivalent to 80 epochs)
    """
    train(model, train_data, val_data, decode, checkpoint_dir)

    # ===========================================================================
    # SAVE METADATA (FOR INFERENCE)
    # ===========================================================================

    """
    Save encode/decode functions so inference can use them.

    In production, you'd also save:
    - Full model config (architecture hyperparameters)
    - Training state (for resuming training)
    - Metadata (dataset info, training duration, etc.)
    """
    import pickle
    meta_path = os.path.join(checkpoint_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump({'encode': encode, 'decode': decode}, f)
    print(f"Saved metadata to {meta_path}")
