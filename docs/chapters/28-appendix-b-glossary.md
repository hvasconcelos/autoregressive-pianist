# Appendix B. Glossary

| Term | Meaning |
|----|----|
| Attention | The part of a transformer that lets each position draw information from earlier positions. |
| Augmentation | Altering training examples in harmless ways (transposing, stretching) to create variety. |
| Autoregressive | Producing output one piece at a time, each chosen in light of all the previous ones. |
| Batch | A group of training sequences processed together in one step. |
| Causal mask | The rule that a position may not look at later positions. |
| Checkpoint | A saved copy of the model's parameters. |
| Context (window) | The tokens the model can see at once: 1,024 here. |
| Cross-entropy | The loss used for training: minus the log of the probability given to the correct token. |
| Embedding | The learned vector that stands for a token inside the model. |
| Fine-tuning | Continuing the training of an already trained model on new data. |
| Gradient | For each parameter, the direction and amount to change it to reduce the loss. |
| Key/value cache | Stored attention results for earlier tokens, so generation only processes the newest one. |
| Logits | The model's raw scores for each possible next token, before they become probabilities. |
| Lookahead | How far ahead of the clock, in seconds, the player keeps notes generated. |
| MIDI | The standard format for note events sent between musical devices. |
| MLX | Apple's machine-learning framework for Apple silicon. |
| Overfitting | Memorising the training data at the expense of doing well on new data. |
| Parameter | One of the learned numbers that make up a model. |
| Perplexity | e to the power of the loss: the effective number of choices the model is torn between. |
| RoPE | Rotary position embedding: a way to tell attention how far apart two tokens are. |
| Tag | A label from a fixed list that describes a passage, used as the request in Part I. |
| Temperature | A sampling setting; higher values make choices more random. |
| Token | One symbol of the model's vocabulary. |
| Top-p | A sampling setting that discards the least likely tokens. |
| Underrun | A moment when the generator fails to produce a note before it is due. |
| Validation set | Data held back from training, used to measure real progress. |
| Velocity | How hard a key is struck, 1 to 127. |
| Weight tying | Using the embedding table as the output layer as well. |
