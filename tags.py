"""Compute the request tags of a passage directly from its notes."""
# numpy: array maths for the pitch histogram, correlations and threshold lookups
import numpy as np
# KEYS: the 24 key names ("Cmaj".."Bmin"); TAG_VALUES: allowed values for each tag category
from tokenizer import KEYS, TAG_VALUES

# Krumhansl-Kessler key profiles: how strongly each of the 12 pitch classes
# belongs to a major / minor key whose tonic is pitch class 0.
# Index 0 = the tonic (C for C major), 7 = the fifth (G); high numbers = notes that "fit" the key.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
# same for minor keys; note index 3 (the minor third, e.g. Eb in C minor) is strong here
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# bucket boundaries: 4 edges make 5 buckets (very_sparse, sparse, medium, dense, very_dense)
DENSITY_EDGES = [4, 7, 11, 16]       # notes per second
# 4 edges -> 5 loudness buckets (pp, p, mf, f, ff); MIDI velocity runs 1-127.
DYNAMICS_EDGES = [45, 58, 70, 82]    # mean MIDI velocity
# 2 edges -> 3 register buckets (low, mid, high).
REGISTER_EDGES = [57, 69]            # mean MIDI pitch (57 = A3, 69 = A4)

# composer surname (lower-case, accent-free) -> musical era. Used to tag pieces by era and
# composer from MAESTRO's composer names; surnames are matched as whole words.
ERA = {"baroque": ["bach", "handel", "scarlatti", "rameau", "couperin", "purcell",
                   "fischer"],
       # roughly 1750-1820
       "classical": ["haydn", "mozart", "beethoven", "clementi", "soler"],
       # roughly 1820-1900
       "romantic": ["schubert", "chopin", "schumann", "liszt", "mendelssohn",
                    "brahms", "grieg", "tchaikovsky", "franck", "balakirev",
                    "mussorgsky", "wagner", "verdi", "glinka", "weber", "bizet",
                    "strauss", "rimsky"],
       # roughly 1890 onwards (late romantic, impressionist and 20th century)
       "modern": ["rachmaninoff", "scriabin", "debussy", "ravel", "prokofiev",
                  "bartok", "berg", "medtner", "janacek", "albeniz", "stravinsky",
                  "shostakovich", "busoni", "kapustin", "szymanowski"]}


def estimate_key(notes):
    """Krumhansl-Schmuckler: correlate the passage's pitch-class histogram
    (weighted by duration) with the 24 rotated key profiles."""
    # one bin per pitch class (C, C#, D, ... B), ignoring which octave a note is in.
    hist = np.zeros(12)
    # add each note's duration (capped at 2 s so one long held note can't dominate) into the
    # bin of its pitch class (pitch % 12, e.g. 60 and 72 are both C -> bin 0). np.add.at is
    # used instead of hist[idx] += w because it accumulates correctly when idx repeats.
    np.add.at(hist, notes[:, 1].astype(int) % 12, np.minimum(notes[:, 3], 2000))
    # best key so far and its correlation (-2 is below any possible correlation, -1..1)
    best, best_r = None, -2.0
    # mode 0 = major profile, mode 1 = minor profile
    for mode, profile in enumerate((MAJOR, MINOR)):
        # try each of the 12 possible tonics (0 = C, 1 = Db, ... 11 = B).
        for tonic in range(12):
            # np.roll shifts the C-based profile right by `tonic` steps, giving the profile of
            # the key on that tonic (roll by 2 -> D major); then take the Pearson correlation.
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            # keep the key whose profile best matches the histogram
            if r > best_r:
                # KEYS lists the 12 majors then the 12 minors, in the same tonic order.
                best, best_r = KEYS[mode * 12 + tonic], r
    # name of the best-matching key, e.g. "Dmin"
    return best


def compute_tags(notes):
    """Tags that can be measured from the notes of one passage.

    Each measurement is bucketed by the *_EDGES thresholds above; the
    bucket index picks the tag value."""
    # passage length in seconds, from the first to the last onset (column 0, in ms)
    span_s = max((notes[-1, 0] - notes[0, 0]) / 1000.0, 1.0)   # at least 1 s
    # notes per second -> bucket index: searchsorted finds how many edges the value is >= to
    # (side="right"), e.g. 8 notes/s with edges [4, 7, 11, 16] -> 2 ("medium").
    d = int(np.searchsorted(DENSITY_EDGES, len(notes) / span_s, side="right"))
    # mean velocity (column 2) -> loudness bucket index
    v = int(np.searchsorted(DYNAMICS_EDGES, notes[:, 2].mean(), side="right"))
    # mean pitch (column 1) -> register bucket index
    r = int(np.searchsorted(REGISTER_EDGES, notes[:, 1].mean(), side="right"))
    # turn each bucket index into its tag name and add the estimated key
    return {"density": TAG_VALUES["density"][d],
            "dynamics": TAG_VALUES["dynamics"][v],
            "register": TAG_VALUES["register"][r],
            "key": estimate_key(notes)}


def composer_tags(name):
    """'Frédéric Chopin' -> {'composer': 'chopin', 'era': 'romantic'}.
    For arrangements ('Franz Schubert / Franz Liszt') the first name wins.
    Composers outside TAG_VALUES get only an era; unknown ones get {}."""
    # re: whole-word search for surnames; unicodedata: removing accents
    import re, unicodedata
    # strip accents so "Frédéric" matches "frederic", then lower-case
    # Keep only the part before "/" (the original composer), then NFKD-decompose it so that
    # accented letters become base letter + separate combining accent ("é" -> "e" + "´").
    plain = unicodedata.normalize("NFKD", name.split("/")[0])
    # drop the combining accent characters and lower-case what remains
    plain = "".join(c for c in plain if not unicodedata.combining(c)).lower()
    # the tags found so far
    out = {}
    # walk every era and every surname listed under it
    for era, names in ERA.items():
        for n in names:
            # \b...\b = whole word only, so "berg" does not match inside "goldberg".
            if re.search(rf"\b{n}\b", plain):
                # any listed composer gives an era tag
                out["era"] = era
                # only composers the tokenizer has a token for also get a composer tag
                if n in TAG_VALUES["composer"]:
                    out["composer"] = n
                # stop at the first match
                return out
    # no known surname found: return the empty dict
    return out
