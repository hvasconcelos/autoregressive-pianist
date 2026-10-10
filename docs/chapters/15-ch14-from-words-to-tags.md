# 14. From words to tags

The model understands tags. People write sentences. This chapter covers the step between.

## 14.1 The keyword table

The simplest translator is a table of words. It needs no model, runs instantly and never fails in surprising ways.

::: filename
request.py
:::

``` python
"""Turn a request written in plain words into tags.

    python request.py "something slow and quiet in D minor, like Chopin"

Two ways: a keyword table (instant, no model needed) and, with --llm, a
small language model that fills in the tags as JSON.
"""
import json, re, sys
from tokenizer import TAG_VALUES, KEYS

# category -> value -> words that ask for it. Words match at the start of a
# word, so a stem like "melanchol" also catches "melancholy" and "melancholic".
KEYWORDS = {
    "density": {"very_sparse": ["very slow", "very sparse", "minimal", "still"],
                "sparse": ["slow", "sparse", "calm", "gentle", "peaceful", "adagio", "sad",
                           "melanchol"],
                "medium": ["moderate", "andante", "walking"],
                "dense": ["fast", "lively", "busy", "allegro", "energetic", "happy",
                          "cheerful"],
                "very_dense": ["very fast", "virtuos", "furious", "presto", "frantic"]},
    "dynamics": {"pp": ["very quiet", "very soft", "whisper", "pianissimo"],
                 "p": ["quiet", "soft", "gentle", "delicate", "calm", "tender"],
                 "mf": ["moderately loud", "mezzo"],
                 "f": ["loud", "strong", "bold", "forte", "energetic"],
                 "ff": ["very loud", "thunder", "powerful", "fortissimo", "furious"]},
    "register": {"low": ["low", "deep", "dark", "bass"],
                 "mid": ["middle register", "mid register"],
                 "high": ["high", "bright", "sparkl", "treble", "music box"]},
    "era": {"baroque": ["baroque"], "classical": ["classical era", "classical style"],
            "romantic": ["romantic"], "modern": ["modern", "impressionis", "20th"]},
    "genre": {"classical": ["classical"],
              "jazz": ["jazz", "swing", "bebop", "blues", "bossa"],
              "pop": ["pop", "rock"],
              "film": ["film", "movie", "soundtrack", "cinematic"],
              "ragtime": ["ragtime", "stride"],
              "other": ["folk", "ambient"]},
}
# when no key is named, a mood word suggests one
MOOD_KEYS = {"sad": "Amin", "melanchol": "Dmin", "dark": "Cmin",
             "happy": "Cmaj", "cheerful": "Gmaj", "bright": "Dmaj"}
FLATS = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}


def parse_request(text):
    """Keyword matching. Longer phrases win, so 'very slow' beats 'slow'."""
    low = text.lower()
    has = lambda w: re.search(r"\b" + re.escape(w), low)     # match at a word start
    tags = {}
    for cat, table in KEYWORDS.items():
        hits = [(len(w), value) for value, words in table.items() for w in words if has(w)]
        if hits:
            tags[cat] = max(hits)[1]
    # a key: "D minor", "F# major", "in b flat major" (but not "a minor key")
    m = re.search(r"(?:\b[Ii]n\s+([A-Ga-g])|\b([A-G]))[\s-]*(#|b|sharp|flat)?[\s-]*"
                  r"((?i:major|minor|maj|min))\b(?!\s+key)", text)
    if m:
        note = (m.group(1) or m.group(2)).upper()
        acc = (m.group(3) or "").lower()
        note += {"#": "#", "sharp": "#", "b": "b", "flat": "b"}.get(acc, "")
        key = FLATS.get(note, note) + ("min" if m.group(4).lower().startswith("min") else "maj")
        if key in KEYS:
            tags["key"] = key
    else:                                    # no key named: guess from the mood
        for word, key in MOOD_KEYS.items():
            if has(word):
                tags["key"] = key
                break
    for name in TAG_VALUES["composer"]:
        if has(name):
            tags["composer"] = name
    return tags


PROMPT = """You convert a request for piano music into tags. Reply with one JSON
object and nothing else. Use only these keys and values, and leave a key out
if the request says nothing about it.
{options}
Request: {request}
JSON:"""


def parse_request_llm(text, model_name="mlx-community/Qwen3-1.7B-4bit"):
    """Ask a small instruct model for the tags, then keep only valid ones."""
    from mlx_lm import load, generate
    model, tok = load(model_name)
    options = "\n".join(f'"{k}": one of {v}' for k, v in TAG_VALUES.items())
    messages = [{"role": "user", "content": PROMPT.format(options=options, request=text)}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    reply = generate(model, tok, prompt=prompt, max_tokens=120)
    # take the first {...} in the reply; anything unparsable means no tags
    found = re.search(r"\{.*\}", reply, re.S)
    try:
        raw = json.loads(found.group(0)) if found else {}
    except json.JSONDecodeError:
        raw = {}
    return {k: v for k, v in raw.items() if v in TAG_VALUES.get(k, [])}


def parse_tags(spec):
    """'key=Dmin,dynamics=p' -> a checked tags dict."""
    tags = dict(item.split("=") for item in spec.split(",") if item)
    for k, v in tags.items():
        if v not in TAG_VALUES.get(k, []):
            choices = TAG_VALUES.get(k, list(TAG_VALUES))
            raise SystemExit(f"unknown tag {k}={v}; choices: {choices}")
    return tags


if __name__ == "__main__":
    # the rest of the command line is the request; --llm picks the model parser
    args = [a for a in sys.argv[1:] if a != "--llm"]
    fn = parse_request_llm if "--llm" in sys.argv else parse_request
    print(fn(" ".join(args)))
```

`parse_request` looks for each keyword at the start of a word in the request. When several match in one category, the longest wins, so "very slow" beats "slow". A key is found with a pattern that accepts forms such as "D minor", "F# major" and "in b flat major". A lower-case note name counts only after the word "in", so that "a minor change" is not read as the key of A minor. If no key is named, mood words suggest one: "sad" becomes A minor.

Try it:

``` bash
python request.py "something slow and quiet in D minor, like Chopin"
```

``` text
{'density': 'sparse', 'dynamics': 'p', 'key': 'Dmin', 'composer': 'chopin'}
```

``` bash
python request.py "very fast and loud, bright, F# major"
```

``` text
{'density': 'very_dense', 'dynamics': 'f', 'register': 'high', 'key': 'Gbmaj'}
```

Genre works the same way. "Jazzy", "swing" and "blues" ask for jazz; "film", "soundtrack" and "cinematic" for film music; "ragtime" and "stride" for ragtime; "pop" and "rock" for pop; and "classical" on its own for classical. The era table deliberately avoids plain "classical", since people use it for any old music, which is what the genre means.

``` bash
python request.py "a jazzy, slow ballad in F major"
```

``` text
{'density': 'sparse', 'genre': 'jazz', 'key': 'Fmaj'}
```

"Ballad" is not a keyword, because there are ballads in every genre. If it were one for pop, it would beat "jazz" here, because the longest match wins.

Anything the table does not recognise is ignored, and the missing tags are simply left out. Thanks to tag dropout in training, the model handles that well.

## 14.2 Using a language model instead

A keyword table cannot cope with "not too fast" or "something for a rainy Sunday". A small language model can. `parse_request_llm` gives one the list of allowed tags and asks for a JSON object.

To use it, install `mlx-lm` and add `--llm`:

``` bash
uv pip install mlx-lm
python request.py --llm "something for a rainy Sunday afternoon"
```

The last line of the function is the important one. Whatever the language model replies, only keys and values that exist in our tag list are kept. A language model will sometimes invent a tag, and an invented tag has no token.

::: {.admonition .warning}
Not run for this book

This function was written against the `mlx-lm` interface but could not be run on the test machine. The model name in the code, a 4-bit Qwen3 from the `mlx-community` collection, is a suggestion; substitute any small instruction-following model that `mlx-lm` can load.
:::

This approach is a stepping stone. It gives you free-text requests today, with two models running side by side. Part II removes the translation step by training one model that reads the text itself.
