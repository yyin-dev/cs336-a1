This is a project to build and train a language model from scratch. This is
an assignment from a LLM class. The writeup is in ../cs336_spring2025_assignment1_basics.pdf

First I train a BPE tokenizer. Then I tokenizer dataset with it.
After that I train a model. In the end I do inference with the trained model.

Bpe training script: experiments/train_bpe.py
Dataset encoding script: experiments/tokenizer_experiment.py
Training script: experiments/train.py
Inference script: experiments/generate_text.py

There are some unit tests that you can run using `uv run pytest`. I have
alread passed all tests, and trained a model. Command:
```
uv run experiments/train.py \
--train_filename ../a1-data/ts-train-encoded-tiktoken.npy \
--val_filename ../a1-data/ts-valid-encoded-tiktoken.npy \
--vocab_size 10000 \
--num_heads 16 \
--d_model 512 \
--d_ff 1344 \
--context_length 256 \
--num_layers 4 \
--rope_theta 10000 \
--batch_size 64 \
--lr_max 3e-3 \
--save_every_n_iterations 2000 \
--total_tokens_processed 327_000_000 \
--save_checkpoint_path ../a1-checkpoints/weight_tying.pt \
--wandb_run_name weight_tying \
--wandb_proj_name cs336-a1
```

The model has about 17M parameters, and trained on about 300M tokens. The training
obtained good validation loss (around 1.3). The dataset is TinyStories.


However, at inference, the model seems to generate garbage. The best I can do, after sweeping temperature and top-p threshold, is following:

* Prompt: Once upon a time
* Generation: judamond pictures lose shot candyrup benches Melissa thrown lighter sour JumpDaddy dot Lily stomachulpture salefortable Next boot gap flappedjam stopSqueaky cushions timer necklaces cleaner lonheart dent wouldizzie enormousasha twentyun palace SparkleSc taken� mate person mos7Ex ever waters zipp acce hairdress la palmcerard comm spearinnybbed Tomm

The writeup seems to indicate that a validation loss of 1.4 on TS is enough to generate fluent text, and I think I have implemented the inference script correctly...

Observation: The inference output is more English-like with a higher temperature (>3), while garbage with a low temperature (<0.5). 

# Task

Help me debug further: why my model, after trained on 300M tokens on TinyStories and with a good validation loss (< 1.0), still cannot generate fluent text at inference?

Your predecessor has helped me debug a bit. The processes was written down to ./DEBUGGING.md, read it understand the things we have tried.

## Fix 1

We caught a bug with SwiGLU weight initialization. 

I fixed the weight initialization, and retrained the model. This does seem to have an effect:
after training on 327M tokens, the validation loss is ~0.8, compared to ~1.4 with the broken initializaiton. 
At iteration 17999, I get validation loss around 0.8.

The checkpoint was saved to ../a1-checkpoints/fix_swiglu_iter17999.pt

However, I tried inference with that checkpoint, and the model still generates garbage.

# Fix 2

We caught a bug with embedding weight initialization. 

I fixed the weight initialization, and retrained the model. At iteration 15999, I get validation loss around 0.8.

The checkpoint was saved to ../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt

However, I tried inference with that checkpoint, and the model still generates garbage.

# Fix 3

We found the bug! When loading data, we must specify `dtype=np.uint16`!

# Note
- At one point I was not sure if there's a bug my own tokenizer implementation and thus tried tiktoken. Howwever, the encoded train&val data is identical for my tokenizer and tiktoken, so I think the tokenizer implementation itself is correct.
- When you need to run a python script, use `uv run`.
- There are some tests. Run `uv run pytest`
- When testing inferencing, try to generate just a small number of tokens (e.g. 32).