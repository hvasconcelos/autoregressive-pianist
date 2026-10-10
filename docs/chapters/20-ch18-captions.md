# 18. Captions

Qwen is conditioned on text, so every training passage needs a sentence that describes it. Neither dataset has descriptions: MAESTRO has a composer and a title, and Aria-MIDI has labels, not sentences. We write them automatically from the tags we already have.

::: filename
qwen/captions.py
:::

``` python
"""Write a one-sentence description of a passage from its tags.

Qwen is conditioned on plain text, so every training passage needs a
caption. We build one from the same measured tags as in Part I, with
varied wording so the model does not latch on to a single phrasing.
"""
import numpy as np

WORDS = {
    "density": {"very_sparse": ["very slow", "sparse and still", "very spacious"],
                "sparse": ["slow", "calm", "unhurried"],
                "medium": ["moderately paced", "flowing", "mid-tempo"],
                "dense": ["fast", "lively", "busy"],
                "very_dense": ["very fast", "virtuosic", "furious"]},
    "dynamics": {"pp": ["very quiet", "whispered", "pianissimo"],
                 "p": ["quiet", "soft", "gentle"],
                 "mf": ["moderately loud", "mezzo-forte"],
                 "f": ["loud", "strong", "forte"],
                 "ff": ["very loud", "thunderous", "fortissimo"]},
    "register": {"low": ["in the low register", "deep in the bass"],
                 "mid": ["in the middle of the keyboard"],
                 "high": ["high on the keyboard", "in the bright upper register"]},
    "era": {"baroque": ["baroque"], "classical": ["classical-era"],
            "romantic": ["romantic"], "modern": ["early modern"]},
    "genre": {"classical": ["classical"], "jazz": ["jazz", "jazzy"], "pop": ["pop"],
              "film": ["cinematic", "film-score"], "ragtime": ["ragtime"],
              "other": []},
}
# {adj} is the joined adjectives, {Adj} the same capitalised, {A} its article
OPENERS = ["{A} {adj} piano piece", "Piano music, {adj}", "Play something {adj}",
           "{Adj} solo piano", "{A} {adj} performance"]
PLAIN_OPENERS = ["A piano piece", "Solo piano music", "Play the piano", "A piano performance"]

# How each of our 24 key names is usually written. Where two spellings are
# in common use, both are listed and one is picked at random.
MAJOR_NAMES = {"Db": ["D-flat"], "Eb": ["E-flat"], "Gb": ["G-flat", "F-sharp"],
               "Ab": ["A-flat"], "Bb": ["B-flat"]}
MINOR_NAMES = {"Db": ["C-sharp"], "Eb": ["E-flat", "D-sharp"], "Gb": ["F-sharp"],
               "Ab": ["G-sharp", "A-flat"], "Bb": ["B-flat"]}


def key_words(key, rng):
    """'Dbmin' -> 'C-sharp minor'."""
    note, minor = key[:-3], key.endswith("min")
    names = (MINOR_NAMES if minor else MAJOR_NAMES).get(note, [note])
    return f"{names[rng.integers(len(names))]} {'minor' if minor else 'major'}"


def caption(tags, rng):
    """tags dict -> a sentence. Missing tags are simply not mentioned."""
    pick = lambda options: options[rng.integers(len(options))]
    keys = ["density", "dynamics", "era", "genre"]
    if "era" in tags and tags.get("genre") == "classical":
        keys.remove("genre")                 # an era already says classical
    adjs = [pick(WORDS[k][tags[k]]) for k in keys
            if WORDS[k].get(tags.get(k))]           # "other" genre has no words
    rng.shuffle(adjs)                        # vary the word order too
    if adjs:
        adj = ", ".join(adjs)
        article = "An" if adj[0] in "aeiou" else "A"   # good enough for these words
        text = pick(OPENERS).format(adj=adj, Adj=adj[0].upper() + adj[1:], A=article)
    else:
        text = pick(PLAIN_OPENERS)
    if "register" in tags:
        text += ", " + pick(WORDS["register"][tags["register"]])
    if "key" in tags:
        text += f", in {key_words(tags['key'], rng)}"
    if "composer" in tags:
        text += pick([f", in the style of {tags['composer'].title()}",
                      f", like {tags['composer'].title()}"])
    return text + "."


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    full = {"density": "sparse", "dynamics": "p", "register": "low", "key": "Dbmin",
            "era": "romantic", "composer": "chopin"}
    for _ in range(4):
        print(caption(full, rng))
    print(caption({"era": "modern", "key": "Bbmaj"}, rng))
    print(caption({"density": "dense", "dynamics": "f", "genre": "jazz"}, rng))
    print(caption({}, rng))
```

For each passage, `caption` picks one wording at random for each tag present (the genre becomes an adjective such as "jazzy" or "cinematic"; `other` has no wording and is left out, and so is `classical` when an era is present, since "classical, romantic" says the same thing twice), shuffles the adjectives, picks one of five sentence openings, and adds the register, key and composer when present. Running the file shows some examples:

``` bash
python qwen/captions.py
```

``` text
Piano music, romantic, unhurried, soft, in the low register, in C-sharp minor, in the style of Chopin.
A quiet, romantic, slow performance, deep in the bass, in C-sharp minor, like Chopin.
Piano music, unhurried, romantic, gentle, deep in the bass, in C-sharp minor, like Chopin.
A slow, romantic, soft performance, in the low register, in C-sharp minor, in the style of Chopin.
An early modern performance, in B-flat major.
Play something jazz, strong, fast.
Solo piano music.
```

## 18.1 Why the wording varies

If every caption were built the same way, such as "A slow, quiet piece in D minor.", the model would learn that exact template and be thrown by anything else. Varying the vocabulary and the order pushes it to rely on what the words mean, which Qwen's pre-training already supplies. A request phrased in a way that never appeared in training has a fair chance of working because "unhurried" and "slow" are already close together inside the model.

The same tag dropout as in Part I applies, so many captions mention only one or two properties and one in ten says nothing beyond "a piano piece". That teaches the model to cope with short requests.

Keys are written the way musicians usually write them: C-sharp minor and not D-flat minor. Where two spellings are both common, such as F-sharp and G-flat major, one is picked at random.

## 18.2 The limits of generated captions

Be clear about what this does and does not achieve. The captions only ever describe the seven tags. Qwen's language ability makes the model flexible about *how* those properties are asked for. It does not teach the model properties that no caption mentions. A request for "a waltz" will not produce three-four time, because no training caption said which passages were waltzes.

There are two ways to widen the range later.

**Richer captions for MAESTRO.** MAESTRO's metadata includes each piece's title, such as "Nocturne in E-flat major" or "Etude Op. 10 No. 4". A language model can turn titles into extra caption words (nocturne, étude, sonata, waltz). This is a small change to `make_batch` and a large gain in vocabulary. Aria-MIDI's metadata already has a `form` label (waltz, nocturne, sonata and so on) for 60,848 recordings, which could go straight into the captions.

**A captioned dataset.** MidiCaps provides 168,000 MIDI files with written descriptions that cover genre, mood, tempo and instrumentation. Its files are multi-instrument band arrangements and not piano performances, so they would need to be reduced to a single piano part and would bring score-like timing. Treat it as a second-stage experiment.
