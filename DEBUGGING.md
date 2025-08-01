# Language Model Inference Debugging Log

## High-Level Investigation Summary

**FINAL ROOT CAUSE**: Data loading bug - training script loaded `uint16` encoded data as `uint8`, causing 84% of tokens to be truncated from meaningful BPE tokens to random bytes.

### Investigation Phases Overview

| Phase | Hypothesis | Validation Process | Result | Next Step |
|-------|------------|-------------------|---------|-----------|
| **1** | Tokenizer/architecture bugs | Compare tokenizers, analyze model weights/activations | ❌ Architecture was correct | Check initialization |
| **2** | Weight initialization issues | Fix SwiGLU (900x too large) and embedding (22x too large) | ⚠️ Training improved but still garbage generation | Investigate training dynamics |
| **3** | Training process bias | Analyze token prediction distributions, embedding norms | ❌ Misdiagnosed symptoms as training bias | Check data pipeline |
| **4** | Data loading bug | Check actual training data vs encoded data | ✅ **BREAKTHROUGH**: `uint8` vs `uint16` mismatch | Fix and retrain |

**Key Insight**: Even with perfect model architecture and initialization, a single-line data loading bug can completely corrupt training by making the model learn on the wrong data distribution.

---

## Problem Statement
- Model trained with ~17M parameters on 300M tokens of TinyStories dataset
- Achieved good validation loss (~1.3) during training
- At inference, generates garbage text instead of fluent English
- Other students with similar validation loss achieved fluent generation

**Example Output:**
```
Prompt: "Once upon a time"
Generation: "judamond pictures lose shot candyrup benches Melissa thrown lighter..."
```

## Environment Details
- Model: 4-layer transformer, 16 heads, d_model=512, d_ff=1344, vocab_size=10000
- Training checkpoint: `../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt` (iteration 18999)
- BPE tokenizer: `../a1-log/ts-bpe.pkl`
- Observation: Higher temperature (>3) produces more English-like text

---

# Phase 1: Initial Investigation

## Initial Hypotheses (Ruled Out)
1. **Tokenizer Mismatch** ❌
   - **Hypothesis**: Training used different tokenizer than inference
   - **Investigation**: Compared custom BPE vs tiktoken encoding
   - **Result**: Both produce identical token sequences
   - **Files**: `debug_tokenizer.py`, `experiments/tiktoken_experiment.py`

2. **Weight Tying Issues** ❌
   - **Hypothesis**: Missing weight tying between input/output embeddings
   - **Investigation**: Git history showed weight tying was never implemented
   - **Result**: Both training and inference use separate embeddings consistently

3. **Cross-Entropy Loss Bug** ❌
   - **Investigation**: Previous bug was fixed in commit b9d9878
   - **Result**: Current implementation is mathematically correct

## Deep Model Analysis
4. **Token Generation Pattern Discovery** ⚠️
   - **Investigation**: `debug_inference.py` revealed model strongly predicts low token IDs
   - **Findings**:
     - Top predictions: tokens 0-26 (individual bytes)
     - Token 24: '', Token 181: '�', Token 34: '"'
     - Model generates individual bytes rather than meaningful subwords
   - **Significance**: This pointed to fundamental training instability

5. **Forward Pass Analysis** 🔍
   - **Investigation**: `debug_shapes.py` traced activations through each layer
   - **Critical Finding**: Massive activation explosion after transformer blocks
   ```
   After input embedding: mean=-0.004, std=0.364
   After transformer block 0: mean=-147562, std=7263663  ⚠️ EXPLOSION
   After transformer block 1: mean=-138387, std=9192264
   After layer norm: mean=0.005, std=0.364  (rescaled back)
   ```

6. **Weight Statistics Analysis** 🔍
   - **Investigation**: `debug_weights.py` and `debug_attention.py`
   - **Key Findings**:
     - Input embedding: std=0.364 (reasonable)
     - Output embedding: std=0.047 (reasonable)
     - **Attention weights**: First block trained properly, others barely updated
   ```
   Block 0: W_Q std=0.164, W_K std=0.188, W_V std=0.211  (learned)
   Block 1: W_Q std=0.016, W_K std=0.016, W_V std=0.016  (barely changed)
   Block 2-3: Similar to block 1 (barely changed)
   ```

## Root Cause Identification (First Bug)
7. **SwiGLU Investigation** ✅ **FIRST ROOT CAUSE FOUND**
   - **Location**: `src/swiglu.py:10`
   - **Bug**: `std = math.sqrt((d_model + d_ff) / 2)`
   - **Analysis**:
     - For d_model=512, d_ff=1344: std = √(1856/2) = **30.46**
     - Correct: `std = math.sqrt(2 / (d_model + d_ff))` = **0.033**
     - **900x larger initialization than intended**
   
   - **Evidence from `debug_attention.py`**:
   ```
   FFN output: mean=-147562, std=7263663
   FFN extreme values: min=-30106724, max=23372928  ⚠️ CATASTROPHIC
   ```

## Technical Analysis (Phase 1)

### Why Validation Loss Looked Good
- Layer norm after each block rescaled extreme values back to ~0.36 std
- Model learned basic byte-level patterns despite instability
- Cross-entropy loss can appear reasonable (~1.3) even with poor text quality
- The rescaling masked the severity of the underlying training dynamics

### Why Only First Block Learned
- Massive gradient explosion from SwiGLU prevented effective learning in deeper layers
- First block received more direct gradient signal from the loss
- Blocks 2-4 weight statistics (std ~0.016) are very close to initialization values

### Why Higher Temperature Helps
- Degenerate model strongly favors low token IDs (bytes)
- Higher temperature reduces this bias, allowing higher token IDs (actual words)
- This is a symptom, not a solution - the fundamental model is broken

## Solution (Phase 1)

### Immediate Action Required
1. ✅ **Fixed SwiGLU initialization** in `src/swiglu.py:10`
2. **Retrain model** - existing checkpoint cannot be salvaged
3. The current weights were optimized under fundamentally broken dynamics

### Code Fix Applied
```python
# OLD (buggy):
std = math.sqrt((d_model + d_ff) / 2)  # std = 30.46

# NEW (correct):
std = math.sqrt(2 / (d_model + d_ff))  # std = 0.033
```

### Verification for Future Training
Monitor these metrics during training:
- Activation statistics through transformer blocks (should stay reasonable)
- Weight update magnitudes across all layers (should be similar)
- Gradient norms (should not explode)

## Files Created During Debugging
- `debug_tokenizer.py` - Tokenizer consistency verification
- `debug_inference.py` - Token generation pattern analysis
- `debug_weights.py` - Model weight statistics
- `debug_shapes.py` - Step-by-step forward pass tracing
- `debug_attention.py` - Detailed attention computation analysis
- `test_buggy_swiglu.py` - SwiGLU behavior investigation

## Key Insights from Phase 1
When validation loss looks reasonable but inference quality is poor, investigate:
1. **Activation statistics** through the forward pass
2. **Weight update patterns** across all layers during training  
3. **Initialization scales** - subtle bugs here can cause cascading failures
4. **Layer norm effects** - can mask underlying instability in loss curves

The bug was a single line of incorrect initialization that caused system-wide training instability, demonstrating how critical proper scaling is in deep learning.

---

# Phase 2: Retrained Model Still Fails

### Retraining Results (Post-SwiGLU Fix)
- **Checkpoint**: `../a1-checkpoints/fix_swiglu_iter17999.pt`
- **Validation loss**: Improved from ~1.4 to ~0.8 ✅
- **Generation quality**: Still produces garbage ❌

## Deeper Investigation

8. **Activation Patterns in Retrained Model** 
   - **Finding**: Activation explosion still occurring despite SwiGLU fix
   - **Evidence**: Values reach -522 to +64 by final block
   - **Token bias**: Still 92.7% probability mass on tokens 0-99
   - **Investigation**: `debug_retrained_model.py`

9. **Weight Comparison Between Checkpoints** 🔍
   - **Shocking discovery**: Retrained model performs WORSE than buggy one
   - **Old model**: 37% probability on low tokens, some actual learning in block 0
   - **New model**: 93% probability on low tokens, appears undertrained
   - **Evidence**: `compare_checkpoints.py`

## Token Distribution Analysis

10. **Model vs Training Data Mismatch** ✅ **SECOND ROOT CAUSE FOUND**
    - **Training data distribution** (correct):
      - Most frequent: `.` (7.7%), `,` (4.3%), ` the` (3.9%)
      - High-quality BPE tokens: ` whiskers`, ` nicest`, ` bold`
    
    - **Model predictions** (wrong):
      - Most probable: `\x18` (14.8%), `\xb5` (10.3%), control characters
      - **Complete failure to learn actual token distribution**
    
    - **Validation paradox explained**: Model confidently predicts wrong tokens
      - Better at byte tokens (loss=7.43) than word tokens (loss=12.89)
      - Cross-entropy rewards confidence, even when wrong

11. **Gradient Flow Analysis** 🔍
    - **Block 0**: Gradient norm = 6,765 (learns significantly)
    - **Blocks 1-3**: Gradient norms = 0.000002-0.000005 (vanishing gradients)
    - **Evidence**: Million-fold difference in gradient magnitudes
    - **Result**: Only first block learns, others remain near initialization

## Second Initialization Bug

12. **Input Embedding Analysis** ✅ **CRITICAL BUG #2 FOUND**
    - **Location**: `src/embedding.py:21`  
    - **Bug**: `std=1.0` (22.6x too large!)
    - **Correct**: `std=1/√d_model = 0.044194`
    - **Impact**: Massive initial activations → training instability
    - **Fix Applied**: Updated to proper scaling

## Updated Technical Analysis

### Why Both Models Failed
1. **SwiGLU bug alone wasn't enough** - embedding bug remained
2. **Cascading instability**: Large embeddings → unstable training → wrong patterns learned
3. **Vanishing gradients**: Deeper blocks (1-3) receive gradients 1M times smaller than block 0
4. **Degenerate learning**: Model learns byte-level patterns instead of word patterns

### The Validation Loss Paradox
- **Key insight**: Cross-entropy loss = -log(p_correct), only cares about probability on correct token
- Model can assign reasonable probability (10-30%) to correct tokens → loss ~1.0-3.0
- But still assigns HIGHEST probability to wrong tokens → bad generation
- **Generation vs Validation**: Validation averages over correct tokens, generation samples from top of distribution
- Layer normalization masks activation explosions in loss curves

### Evidence Summary
**Gradient Flow:**
```
Block 0 attention: 6,765 gradient norm → LEARNS (weights grow 3.7-4.3x)
Block 1 attention: 0.000005 gradient norm → BARELY LEARNS (weights shrink to 0.36x)
Block 2-3: Similar vanishing gradients → NO MEANINGFUL LEARNING
```

**Token Prediction Mismatch:**
```
Training data: . (7.7%), , (4.3%), " the" (3.9%)  ← CORRECT
Model predicts: \x18 (14.8%), \xb5 (10.3%), " (4.9%)  ← WRONG
```

## Complete Solution

### All Fixes Applied ✅
1. **SwiGLU initialization**: `std = √(2/(d_model + d_ff))` = 0.033
2. **Input embedding initialization**: `std = 1/√d_model` = 0.044194
3. **Training setup verified**: Gradient clipping, cosine scheduling all correct

### Files Created in Phase 2
- `debug_retrained_model.py` - Analysis of retrained model activations
- `compare_checkpoints.py` - Weight comparison between old/new models  
- `check_vocab_consistency.py` - Training data and vocab verification
- `debug_generation_step_by_step.py` - Token-level generation analysis
- `analyze_weight_init.py` - Comprehensive initialization analysis
- `analyze_learning_evidence.py` - Gradient flow and learning verification
- `analyze_validation_loss_paradox.py` - Validation loss vs generation quality

---

# Phase 3: Full Training Analysis

## Post-Fix Results
- **Checkpoint**: `../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt`
- **Training dynamics**: ✅ FIXED - All blocks learning, stable activations
- **Generation quality**: ❌ STILL GARBAGE - 99.6% probability on byte tokens

## Improved Training Dynamics Evidence

**Gradient Flow (Post Both Fixes):**
```
Block 0 attention W_Q grad norm: 1.083    → HEALTHY GRADIENTS ✅
Block 1 attention W_Q grad norm: 0.150    → HEALTHY GRADIENTS ✅  
Block 2 attention W_Q grad norm: 0.277    → HEALTHY GRADIENTS ✅
Block 3 attention W_Q grad norm: 0.093    → HEALTHY GRADIENTS ✅

Block 0 SwiGLU W1 grad norm: 4.100       → HEALTHY GRADIENTS ✅
Block 1 SwiGLU W1 grad norm: 3.939       → HEALTHY GRADIENTS ✅
Block 2 SwiGLU W1 grad norm: 3.688       → HEALTHY GRADIENTS ✅  
Block 3 SwiGLU W1 grad norm: 6.942       → HEALTHY GRADIENTS ✅
```

**Weight Learning Evidence:**
```
ATTENTION WEIGHTS (all blocks learning moderately):
Block 0: W_Q std=0.061 (1.38x init), W_K std=0.067 (1.52x init)
Block 1: W_Q std=0.062 (1.41x init), W_K std=0.066 (1.50x init)  
Block 2: W_Q std=0.066 (1.49x init), W_K std=0.069 (1.55x init)
Block 3: W_Q std=0.063 (1.42x init), W_K std=0.066 (1.49x init)

SWIGLU WEIGHTS (all blocks learning significantly):
Block 0: W1 std=0.054 (1.64x init) → LEARNED SIGNIFICANTLY
Block 1: W1 std=0.075 (2.30x init) → LEARNED SIGNIFICANTLY
Block 2: W1 std=0.079 (2.40x init) → LEARNED SIGNIFICANTLY  
Block 3: W1 std=0.080 (2.43x init) → LEARNED SIGNIFICANTLY
```

**Comparison with Phase 2 (Pre-Embedding Fix):**
```
BEFORE (Phase 2): Block 0 grad=6,765, Blocks 1-3 grad=0.000002-0.000005 (million-fold difference)
AFTER (Phase 3):  Block 0 grad=1.08, Blocks 1-3 grad=0.09-0.28 (10x difference, healthy range)
```

**Key Improvement**: Fixed initialization eliminated the catastrophic gradient differences. All blocks now show healthy gradient flow and weight updates, proving the training dynamics were completely fixed. However, they were all learning truncated byte data instead of actual BPE tokens due to the data loading bug.

## Root Cause Discovery (Still Wrong)

## Critical Investigation ❌ **INCORRECT ROOT CAUSE**
13. **Training Process Systematic Bias**
    - **Fresh model initialization**: Perfectly balanced embeddings (ratio=1.00)
    - **Post-training embeddings**: Massive bias - byte tokens 3.4x larger norms
    - **Evidence**: Training process systematically favors low token IDs (0-255)
    
    **Key Findings:**
    ```
    Fresh model: All tokens ~0.99 norm (balanced)
    Trained model: Bytes 2.51 norm, Words 0.73 norm (3.4x bias)
    Model assigns 0% probability to common word " the"
    Training data contains 22% word tokens, 0% control bytes
    ```

## The Misdiagnosed Problem
**We incorrectly believed the training process had a systematic bug favoring low token IDs, but this was actually a symptom of data corruption.**

This explains everything:
- ✅ Model architecture is correct
- ✅ Initialization is correct  
- ✅ Training hyperparameters are correct
- ❌ **Training loop systematically breaks the model**

## Status: MISDIAGNOSED - SYMPTOMS IDENTIFIED
**INCORRECT DIAGNOSIS: Systematic bias in training loop** ❌
- Model initialization: Perfect ✅
- Training dynamics: Appeared to favor byte tokens over word tokens
- **Actual issue**: Data corruption made it appear as training bias

## Key Lessons from Phase 3 (Later Proven Incorrect)
1. **Perfect metrics can hide systematic bias**: Model can have excellent training dynamics while learning completely wrong patterns ✅ (Still valid)
2. **Token-level analysis is crucial**: Must verify the model learns correct token distributions, not just reasonable loss ✅ (Still valid)
3. **Initialization vs training separation**: Even perfect initialization can be destroyed by biased training ❌ (Was data corruption, not training bias)
4. **Embedding norm analysis reveals hidden bias**: Systematic differences in learned embedding magnitudes expose training bugs ⚠️ (Reveals data corruption symptoms)
5. **Fresh model comparison essential**: Comparing trained vs fresh models reveals training-induced bias ⚠️ (Reveals data corruption effects)

**Important Note**: This phase incorrectly attributed the byte bias to training process bugs, when it was actually caused by data corruption (discovered in Phase 4).

---

# Phase 4: The True Root Cause - BREAKTHROUGH DISCOVERY ✅

## Final Investigation: Data Loading Bug

## Direct Training Data Analysis

14. **Training Data Token Distribution Analysis** 🔍
    - **Hypothesis**: Check what tokens the model actually sees during training
    - **Investigation**: Analyzed first 1M tokens from training data as loaded by training script
    - **Shocking Discovery**: Training data contained ONLY bytes (tokens 0-255)!
    
    **Key Findings:**
    ```
    Training data (as loaded): 256 unique tokens, 100% bytes (0-255)
    Expected BPE data: 10,000 unique tokens (0-9999), with 84% being BPE tokens (256+)
    
    Top "training" tokens: \x01 (24.77%), \x00 (8.37%), \x02 (6.05%)  ← WRONG!
    Expected top tokens: . (7.7%), , (4.3%), " the" (3.9%)  ← RIGHT!
    
    Common words like " the" (token 263): 0 occurrences in "training" data
    ```
    
    - **Evidence**: `debug_training_data.py`

15. **Data Loading Investigation** ✅ **TRUE ROOT CAUSE FOUND**
    - **Hypothesis**: Training script incorrectly loads encoded data
    - **Investigation**: Compare `np.load()` vs `np.memmap()` on encoded files
    - **Critical Bug Found**: 
    
    ```python
    # ENCODED DATA (correct):
    data = np.load("../a1-data/ts-train-encoded-tiktoken.npy")  
    # Shape: (541,229,347,), dtype: uint16, range: 9-9999 ✅
    
    # TRAINING SCRIPT (broken):
    train_dataset = np.memmap(train_filename, mode="r")  # Defaults to uint8!
    # Shape: (1,082,458,822,), dtype: uint8, range: 0-255 ❌
    ```
    
    - **The Bug**: `memmap` defaults to `uint8`, truncating all `uint16` values to 0-255
    - **Impact**: 84% of tokens corrupted from meaningful BPE tokens to random bytes
    - **Evidence**: `debug_data_types.py`, `verify_data_fix.py`

## The Complete Picture

### What Actually Happened
1. **Encoding**: ✅ BPE tokenizer correctly encoded text into tokens 9-9999
2. **Storage**: ✅ Data correctly saved as `uint16` numpy arrays  
3. **Loading**: ❌ Training script loaded as `uint8`, truncating the data  
4. **Training**: ❌ Model learned byte patterns from truncated data
5. **Validation**: ❌ Loss looked good because calculated on same truncated data
6. **Inference**: ❌ Model generates bytes instead of words

### Why All Previous Fixes Failed
- **SwiGLU & Embedding fixes**: Improved training dynamics but couldn't fix corrupted data
- **Training process analysis**: Revealed symptoms (byte bias) but not the cause
- **Model architecture**: Was always correct, never the problem
- **The real issue**: Model never saw actual language tokens during training

## Final Solution ✅

### Code Fix Applied
```python
# BROKEN (experiments/train.py:158):
train_dataset = np.memmap(train_filename, mode="r")  # uint8 default

# FIXED (experiments/train.py:157):
train_dataset = np.memmap(train_filename, mode="r", dtype=np.uint16)
```

### Verification
- ✅ Fixed training data loading shows 90% BPE tokens (vs 0% before)
- ✅ Model now sees proper token distribution during training
- ✅ After retraining with fixed data loading: **INFERENCE WORKS!**

## Final Technical Analysis

### The Data Corruption Chain
```
Original text: "Once upon a time there was"
↓ BPE encoding
Correct tokens: [430, 439, 259, 398, 401, 283]  (meaningful words)
↓ uint8 truncation  
Corrupted tokens: [78, 85, 77, 80, 89, 1]      (random bytes)
```

### Why Validation Loss Was Misleading
- Cross-entropy loss on corrupted data can still reach ~0.8-1.3
- Model learned byte-level patterns that had statistical structure
- Layer normalization masked the fundamental data corruption
- **Key lesson**: Good loss ≠ learning the right distribution

### Files Created in Phase 4
- `debug_training_data.py` - Analysis of actual training data distribution  
- `debug_data_types.py` - Comparison of loading methods
- `verify_data_fix.py` - Verification that fix works
- `debug_encoding_process.py` - BPE vocab consistency check
- `compare_encoded_files.py` - File integrity verification

## Ultimate Key Lessons Learned

1. **Data integrity is paramount**: Always verify the model sees the data you think it does
2. **dtype matters**: A single missing `dtype=np.uint16` corrupted months of work
3. **Validate the training pipeline**: Check data loading, not just model architecture
4. **Symptoms vs root cause**: Byte bias was a symptom of data corruption, not training bias
5. **Loss can be deceiving**: Good metrics on wrong data are worse than bad metrics on right data

**The most subtle bugs can have the most catastrophic effects.** A single line of missing dtype specification corrupted the entire training process, demonstrating the critical importance of end-to-end data pipeline validation in deep learning.