# 22. Comparing the two models

You now have two pianists trained on the same data. Compare them on the same four tests as chapter 15, plus the one thing only Qwen can do.

| Test | How | What to look for |
|----|----|----|
| Loss on unseen music | `evaluate.py` for the small model; the `val` figure in the Qwen log | Lower is better. A difference under about 0.05 is not meaningful. |
| Following the six properties | Tag adherence for the small model; the same requests as sentences for Qwen | Qwen should match the small model here. If it is worse, the captions are the problem. |
| Free-text requests | Twenty sentences in wordings that never appear in `captions.py` | The reason Part II exists. Does "unhurried and hushed" behave like "slow and quiet"? |
| Speed | Underruns in five minutes of dense playing | The small model will win. The question is whether Qwen is fast enough. |
| Listening | Ten files each, same requests, played blind | Which one would you rather listen to? |

To measure how well Qwen follows the six properties, write each tag value as a short sentence, generate with `play_qwen.py --out`, load the file with `load_notes` and measure it with `compute_tags`, exactly as `evaluate.py` does for the small model.

## 22.1 Reading the outcome

**Qwen has lower loss and handles free text.** It is the better model. Use it if it is fast enough, and keep the small one as a fallback.

**The losses are about equal, and Qwen handles free text.** This is the most likely result. The pre-training did not improve the playing, as chapter 16 predicted, but the language interface works. Choose by speed and convenience.

**Qwen is worse at the music.** It is probably overfitting or under-trained. Check its validation curve. A larger model needs more data to show its advantage, and MAESTRO alone is small.

**Qwen is too slow.** Keep the small model as the player and use Qwen's understanding of language a different way: as the request translator from chapter 14. That hybrid, a language model that writes tags and a fast specialist that plays them, is a perfectly good final design.
