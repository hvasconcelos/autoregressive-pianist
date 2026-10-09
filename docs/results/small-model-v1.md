# Small model v1: results

Baseline results for the Part I model (`runs/v1/best`). These are the numbers the
Qwen model in Part II has to beat. The tests and pass marks come from chapter 15
(`docs/chapters/16-ch15-does-it-work-validation.md`), and the comparison plan
comes from chapter 22.

## The run

| | |
|---|---|
| Checkpoint | `runs/v1/best`, step 25,500 (also the last step trained) |
| Model | 6 layers, width 512, 8 heads, context 1024, dropout 0.1, 19.1M parameters, vocab 459 |
| Data | MAESTRO v3.0.0 via `prepare.py`: 962 train pieces (5.66M notes, ~21.7M tokens), 137 validation pieces |
| Training | `--batch 8`, lr 3e-4, warm-up 1,000, cosine to 10%, AdamW. Resumed once at step 10,000 and extended past the planned 20,000. |
| Tokens seen | 25,500 × 8 × 1,023 ≈ 209M, about 11 tokens per parameter and about 9.6 passes over the data |
| Hardware | Apple M1 Pro, 16 GB, MLX. About 14.4k tokens/s, roughly 4.9 hours of training. |
| Final log row | train 2.128, val 2.271 (gap 0.14). Validation was still falling slowly (about 0.01 per 1,000 steps) and had not turned. |
| Code | evaluated at commit `06f74db` |

The default `--batch 32` swapped heavily on 16 GB and made no progress, so this
run used batch 8.

## Test 1: loss on unseen music ✅

`evaluate.py --split test` (20 batches of 16, fixed seed):

| Family | Test loss | Perplexity | Validation loss | Perplexity |
|---|---:|---:|---:|---:|
| pitch | 1.782 | 5.9 | 1.980 | 7.2 |
| velocity | 2.012 | 7.5 | 2.046 | 7.7 |
| duration | 2.623 | 13.8 | 2.674 | 14.5 |
| shift | 2.138 | 8.5 | 2.304 | 10.0 |
| **all** | **2.139** | **8.5** | **2.249** | **9.5** |

Pass mark: overall far below 6 and pitch perplexity well under 88. Both hold by
a wide margin. Duration is the hardest family, because durations include the
sustain pedal. The test pieces are somewhat more predictable than the
validation pieces. MAESTRO keeps each piece in one split only, so this isn't
leakage.

## Test 2: tag adherence ✅

`evaluate.py --keys`, 8 generations of 120 notes per tag value, one tag at a
time, fixed seeds. "near" means within one step for ordered tags. For key it
means the same key, the relative major/minor, or a key a fifth away.

| Tag | Exact | Near | Chance (exact) | Chance (near) |
|---|---:|---:|---:|---:|
| density | 48% | 82% | 20% | — |
| dynamics | 68% | 95% | 20% | — |
| register | 62% | 96% | 33% | — |
| key | 23% | 51% | 4% | 17% |

Pass mark: near above 80% for density and dynamics, and register exact well
above chance. All three hold, though density only just clears the mark. Key
has no pass mark: 23% of generations are in the exact key, 28% in a
neighbouring key and 49% in an unrelated key. Each rate comes from 40–192
generations, so allow about ±10%.

## Test 3: real-time headroom (in progress)

```bash
uv run python sample.py --model runs/v1/best --tags density=very_dense --notes 600 --seed 0
```

| Request | Notes | Music | Generated | Needed | Headroom |
|---|---:|---:|---:|---:|---:|
| `density=very_dense` | 600 | 33.2 s (18 notes/s) | 368 tokens/s | 71 tokens/s | **5.2×** ✅ |

Pass mark: at least 3×. This was measured while a `play.py` session was also
running. During training, generation dropped to about 164 tokens/s.

Still to do: 5 minutes of `play.py` with a dense request and 0–1 underruns.

## Test 4: listening (to do)

Ten files across contrasting requests, listened to blind. An early checkpoint
(step 5,000) played back correctly in Oryon Audio Sonora and Ableton Live.

## Comparison sheet for Part II

The tests from chapter 22, with the small model's figures filled in:

| Test | Small model v1 | Qwen |
|---|---|---|
| Loss on unseen music, validation (all) | 2.249 | |
| Loss on unseen music, test (all) | 2.139 | |
| Tag adherence, near (density / dynamics / register / key) | 82% / 95% / 96% / 51% | |
| Tag adherence, exact (density / dynamics / register / key) | 48% / 68% / 62% / 23% | |
| Free-text requests (20 unseen wordings) | keyword table in `request.py` | |
| Speed, very dense (generated vs needed) | 368 vs 71 tokens/s (5.2×) | |
| Underruns in 5 min of dense playing | to do | |
| Listening (10 files, blind) | to do | |

Loss differences under about 0.05 aren't meaningful (chapter 22). Qwen uses
the same music tokens (`qwen/music_lm.py` scores only those rows), so
per-token losses can be compared directly. Chapter 22 takes Qwen's `val`
figure from its training log, which is a validation loss. Compare it with the
small model's validation 2.249, not with the test 2.139.
