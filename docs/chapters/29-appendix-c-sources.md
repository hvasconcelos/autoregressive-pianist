# Appendix C. Sources

**Checked while writing this book** (October 2026):

- MAESTRO dataset page, with the version 3.0.0 statistics, download links and licence: [magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)
- Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset": [arxiv.org/abs/1810.12247](https://arxiv.org/abs/1810.12247)
- Aria-MIDI dataset card: [huggingface.co/datasets/loubb/aria-midi](https://huggingface.co/datasets/loubb/aria-midi)
- MidiCaps dataset card: [huggingface.co/datasets/amaai-lab/MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps)
- Qwen3-0.6B-Base model card and configuration file: [huggingface.co/Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- Qwen3.5-0.8B-Base model card: [huggingface.co/Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- Hawthorne et al., "Sequence-to-Sequence Piano Transcription with Transformers", for folding the sustain pedal into note durations: [arxiv.org/abs/2107.09142](https://arxiv.org/abs/2107.09142)
- A community setup guide for PyTorch on the DGX Spark: [github.com/natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup)
- MLX documentation: [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx/)

**Background, cited from memory and not re-read for this book:**

- Oore et al., "This Time with Feeling: Learning Expressive Musical Performance" (2018). The origin of event-based performance modelling. arXiv 1808.03715.
- Huang et al., "Music Transformer" (2018). Transformers applied to piano performance. arXiv 1809.04281.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). arXiv 2104.09864.
- Krumhansl, *Cognitive Foundations of Musical Pitch* (1990). The key profiles used in `tags.py`.
- Fradet et al., "MidiTok: A Python Package for MIDI File Tokenization" (2021). Includes the TSD encoding.
- Hewitt, "Initializing New Word Embeddings for Pretrained Language Models" (2021). The method Transformers uses when resizing the embedding table.
