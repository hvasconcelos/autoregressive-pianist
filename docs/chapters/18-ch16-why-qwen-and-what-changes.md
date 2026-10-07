# 16. Why Qwen, and what changes

The small model has one clear limitation: it understands 55 tags and nothing else. A request has to be squeezed through a keyword table or a second model before it reaches the pianist.

A language model removes that step. Qwen already knows what "melancholy", "like a lullaby" and "in the style of Chopin" mean as language. If we teach it to continue a sentence with music tokens, the request can be any sentence.

## 16.1 What stays the same

Nearly everything.

| Component | Part I | Part II |
|----|----|----|
| Dataset and note arrays | `prepare.py` | The same files |
| Music tokens | 400 tokens | The same 400, added to Qwen's vocabulary |
| Augmentation and passage cutting | `data.py` | The same code |
| Generation engine | `performer.py` | The same class |
| Real-time scheduler | `play.py` | The same function |
| Conditioning | Tag tokens | A sentence |
| Model | 19 M parameters, from scratch | About 600 M, pre-trained on text |
| Framework and machine | MLX, MacBook | PyTorch, DGX Spark |

## 16.2 What Qwen brings, and what it does not

**It brings language.** The request is tokenised by Qwen's own tokeniser and understood by layers that were trained on a very large amount of text.

**It does not bring music.** Qwen has read *about* music but has never been trained to produce note events. The music tokens are new to it and start with meaningless vectors. Almost all of its musical skill will come from the same MAESTRO data as before. Do not expect the pre-training to make it a better pianist by itself; expect it to make it a better *listener*.

**It costs speed.** The model is about thirty times larger. Chapter 21 deals with keeping it fast enough.

## 16.3 Which Qwen

Use **Qwen3-0.6B-Base**.

| Property | Value |
|----|----|
| Parameters | 0.6 billion, of which 0.44 billion are outside the embedding table |
| Layers | 28 |
| Vector width | 1,024 |
| Attention heads | 16 (8 key/value heads) |
| Vocabulary rows | 151,936 |
| Input and output embeddings | Tied |
| Context length | 32,768 tokens |
| Licence | Apache 2.0 |
| Requires | Transformers 4.51 or later (this book's code was tested on 5.19) |

Three reasons for this choice.

- **It is the smallest model of the Qwen3 family, and it is text-only.** Size decides real-time speed.
- **It is a Base model.** A Base model has only been trained to continue text. The Instruct and chat versions have been further trained to hold conversations, which is of no use here and would have to be unlearned.
- **Its embeddings are tied.** The same table serves as input and output, so each new music token needs only one new row.

::: {.admonition .note}
What about Qwen3.5?

A newer family exists, and its smallest member is Qwen3.5-0.8B. It is a vision-language model with a different internal architecture and is loaded through a different Transformers class. It may work, but it is larger and nothing in this book was checked against it. Get Qwen3-0.6B working first, then treat a newer model as an experiment.
:::
