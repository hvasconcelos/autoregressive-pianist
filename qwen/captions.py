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
