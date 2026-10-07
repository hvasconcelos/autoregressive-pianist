"""Turn a request written in plain words into tags.

    uv run python request.py "something slow and quiet in D minor, like Chopin"

Two ways: a keyword table (instant, no model needed) and, with --llm, a
small language model that fills in the tags as JSON.
"""
# json: parse the LLM's JSON reply; re: keyword/key matching; sys: command-line arguments
import json, re, sys
# TAG_VALUES: allowed values per tag category; KEYS: the 24 valid key names ("Cmaj".."Bmin")
from tokenizer import TAG_VALUES, KEYS

# category -> value -> words that ask for it. Words match at the start of a
# word, so a stem like "melanchol" also catches "melancholy" and "melancholic".
# A word may appear under several categories (e.g. "calm" means both sparse and quiet).
KEYWORDS = {
    # density = how many notes per second (tempo / busyness)
    "density": {"very_sparse": ["very slow", "very sparse", "minimal", "still"],
                "sparse": ["slow", "sparse", "calm", "gentle", "peaceful", "adagio", "sad",
                           "melanchol"],
                "medium": ["moderate", "andante", "walking"],
                "dense": ["fast", "lively", "busy", "allegro", "energetic", "happy",
                          "cheerful"],
                "very_dense": ["very fast", "virtuos", "furious", "presto", "frantic"]},
    # dynamics = loudness, using the musical markings pp (very soft) .. ff (very loud)
    "dynamics": {"pp": ["very quiet", "very soft", "whisper", "pianissimo"],
                 "p": ["quiet", "soft", "gentle", "delicate", "calm", "tender"],
                 "mf": ["moderately loud", "mezzo"],
                 "f": ["loud", "strong", "bold", "forte", "energetic"],
                 "ff": ["very loud", "thunder", "powerful", "fortissimo", "furious"]},
    # register = how high or low on the keyboard the music sits
    "register": {"low": ["low", "deep", "dark", "bass"],
                 "mid": ["middle register", "mid register"],
                 "high": ["high", "bright", "sparkl", "treble", "music box"]},
    # era = style period; "classical" alone is avoided because people use it for any old music
    "era": {"baroque": ["baroque"], "classical": ["classical era", "classical style"],
            "romantic": ["romantic"], "modern": ["modern", "impressionis", "20th"]},
}
# when no key is named, a mood word suggests one
# (minor keys tend to sound sad/dark, major keys happy/bright)
MOOD_KEYS = {"sad": "Amin", "melanchol": "Dmin", "dark": "Cmin",
             "happy": "Cmaj", "cheerful": "Gmaj", "bright": "Dmaj"}
# the tokenizer names black keys with flats only, so convert sharps: "C#" -> "Db", etc
FLATS = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}


def parse_request(text):
    """Keyword matching. Longer phrases win, so 'very slow' beats 'slow'."""
    # lower-case copy for case-insensitive keyword matching
    low = text.lower()
    # has(w): does w occur starting at a word boundary? re.escape makes spaces/symbols literal
    has = lambda w: re.search(r"\b" + re.escape(w), low)     # match at a word start
    # the tags found, category -> value
    tags = {}
    # go through each category's table of value -> trigger words
    for cat, table in KEYWORDS.items():
        # every trigger word present in the text, as (word length, value) pairs
        hits = [(len(w), value) for value, words in table.items() for w in words if has(w)]
        # if anything matched, pick the value of the longest matching word (max compares
        # length first), so "very slow" (9 chars) wins over "slow" (4 chars).
        if hits:
            tags[cat] = max(hits)[1]
    # a key: "D minor", "F# major", "in b flat major" (but not "a minor key")
    # Regex parts (applied to the original text, so letter case matters):
    #   (?:\b[Ii]n\s+([A-Ga-g]) | \b([A-G]))  note letter: after "in " any case (group 1),
    #                                         otherwise only a capital letter (group 2), so
    #                                         the article "a" in "a minor" isn't taken as A
    #   [\s-]*(#|b|sharp|flat)?[\s-]*         optional accidental (group 3), spaces/hyphens ok
    #   ((?i:major|minor|maj|min))\b          the mode, case-insensitive (group 4)
    #   (?!\s+key)                            reject "... minor key" (e.g. "a minor key change")
    m = re.search(r"(?:\b[Ii]n\s+([A-Ga-g])|\b([A-G]))[\s-]*(#|b|sharp|flat)?[\s-]*"
                  r"((?i:major|minor|maj|min))\b(?!\s+key)", text)
    # A key was named.
    if m:
        # the note letter from whichever alternative matched, upper-cased ("d" -> "D")
        note = (m.group(1) or m.group(2)).upper()
        # the accidental, if any, lower-cased ("" when absent)
        acc = (m.group(3) or "").lower()
        # append "#" for sharp or "b" for flat, e.g. "F" + "sharp" -> "F#"
        note += {"#": "#", "sharp": "#", "b": "b", "flat": "b"}.get(acc, "")
        # map sharps to flats ("F#" -> "Gb") and add "min" or "maj" from the mode word
        key = FLATS.get(note, note) + ("min" if m.group(4).lower().startswith("min") else "maj")
        # only keep it if it is one of the 24 keys the model knows (e.g. "Cbmaj" is not)
        if key in KEYS:
            tags["key"] = key
    else:                                    # no key named: guess from the mood
        # first mood word found (in MOOD_KEYS order) decides the key
        for word, key in MOOD_KEYS.items():
            if has(word):
                tags["key"] = key
                break
    # composer: any composer name the tokenizer knows that appears in the text (last one wins)
    for name in TAG_VALUES["composer"]:
        if has(name):
            tags["composer"] = name
    # e.g. {"density": "sparse", "dynamics": "p", "key": "Dmin", "composer": "chopin"}
    return tags


# prompt for the LLM; {options} is filled with the allowed values, {request} with the text
PROMPT = """You convert a request for piano music into tags. Reply with one JSON
object and nothing else. Use only these keys and values, and leave a key out
if the request says nothing about it.
{options}
Request: {request}
JSON:"""


def parse_request_llm(text, model_name="mlx-community/Qwen3-1.7B-4bit"):
    """Ask a small instruct model for the tags, then keep only valid ones."""
    # imported here so mlx_lm is only needed when --llm is actually used
    from mlx_lm import load, generate
    # download (first time) and load the model weights and its tokenizer
    model, tok = load(model_name)
    # one line per category listing its allowed values, e.g. '"dynamics": one of ['pp', ...]'
    options = "\n".join(f'"{k}": one of {v}' for k, v in TAG_VALUES.items())
    # A single-turn chat containing the filled-in prompt.
    messages = [{"role": "user", "content": PROMPT.format(options=options, request=text)}]
    # wrap it in the model's chat format as plain text, ending where the assistant reply
    # starts; enable_thinking=False turns off Qwen3's "thinking" preamble.
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    # let the model write up to 120 tokens of reply
    reply = generate(model, tok, prompt=prompt, max_tokens=120)
    # take the first {...} in the reply; anything unparsable means no tags
    # (greedy .* with re.S spans newlines, from the first "{" to the last "}")
    found = re.search(r"\{.*\}", reply, re.S)
    # parse the JSON object, or use {} if none was found
    try:
        raw = json.loads(found.group(0)) if found else {}
    # malformed JSON: give up and use no tags
    except json.JSONDecodeError:
        raw = {}
    # keep only known categories with allowed values (drops anything the model made up)
    return {k: v for k, v in raw.items() if v in TAG_VALUES.get(k, [])}


def parse_tags(spec):
    """'key=Dmin,dynamics=p' -> a checked tags dict."""
    # split on commas, then each "name=value" on "=", into a dict (empty items skipped)
    tags = dict(item.split("=") for item in spec.split(",") if item)
    # check every tag against the allowed values
    for k, v in tags.items():
        if v not in TAG_VALUES.get(k, []):
            # valid values for a known category, or the list of categories for an unknown one
            choices = TAG_VALUES.get(k, list(TAG_VALUES))
            # stop the program with a helpful message
            raise SystemExit(f"unknown tag {k}={v}; choices: {choices}")
    return tags


# runs only when this file is executed directly, not when imported
if __name__ == "__main__":
    # the rest of the command line is the request; --llm picks the model parser
    args = [a for a in sys.argv[1:] if a != "--llm"]
    # choose the LLM parser if --llm was given, else the keyword parser
    fn = parse_request_llm if "--llm" in sys.argv else parse_request
    # join the words back into one request string, parse it and print the tags dict
    print(fn(" ".join(args)))
