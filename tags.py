"""Compute the request tags of a passage directly from its notes."""
import numpy as np
from tokenizer import KEYS, TAG_VALUES

# Krumhansl-Kessler key profiles: how strongly each of the 12 pitch classes
# belongs to a major / minor key whose tonic is pitch class 0.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

DENSITY_EDGES = [4, 7, 11, 16]       # notes per second
DYNAMICS_EDGES = [45, 58, 70, 82]    # mean MIDI velocity
REGISTER_EDGES = [57, 69]            # mean MIDI pitch (57 = A3, 69 = A4)

ERA = {"baroque": ["bach", "handel", "scarlatti", "rameau", "couperin", "purcell",
                   "fischer"],
       "classical": ["haydn", "mozart", "beethoven", "clementi", "soler"],
       "romantic": ["schubert", "chopin", "schumann", "liszt", "mendelssohn",
                    "brahms", "grieg", "tchaikovsky", "franck", "balakirev",
                    "mussorgsky", "wagner", "verdi", "glinka", "weber", "bizet",
                    "strauss", "rimsky"],
       "modern": ["rachmaninoff", "scriabin", "debussy", "ravel", "prokofiev",
                  "bartok", "berg", "medtner", "janacek", "albeniz", "stravinsky",
                  "shostakovich", "busoni", "kapustin", "szymanowski"]}


def estimate_key(notes):
    """Krumhansl-Schmuckler: correlate the passage's pitch-class histogram
    (weighted by duration) with the 24 rotated key profiles."""
    hist = np.zeros(12)
    np.add.at(hist, notes[:, 1].astype(int) % 12, np.minimum(notes[:, 3], 2000))
    best, best_r = None, -2.0
    for mode, profile in enumerate((MAJOR, MINOR)):
        for tonic in range(12):
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if r > best_r:
                best, best_r = KEYS[mode * 12 + tonic], r
    return best


def compute_tags(notes):
    """Tags that can be measured from the notes of one passage.

    Each measurement is bucketed by the *_EDGES thresholds above; the
    bucket index picks the tag value."""
    span_s = max((notes[-1, 0] - notes[0, 0]) / 1000.0, 1.0)   # at least 1 s
    d = int(np.searchsorted(DENSITY_EDGES, len(notes) / span_s, side="right"))
    v = int(np.searchsorted(DYNAMICS_EDGES, notes[:, 2].mean(), side="right"))
    r = int(np.searchsorted(REGISTER_EDGES, notes[:, 1].mean(), side="right"))
    return {"density": TAG_VALUES["density"][d],
            "dynamics": TAG_VALUES["dynamics"][v],
            "register": TAG_VALUES["register"][r],
            "key": estimate_key(notes)}


def composer_tags(name):
    """'Frédéric Chopin' -> {'composer': 'chopin', 'era': 'romantic'}.
    For arrangements ('Franz Schubert / Franz Liszt') the first name wins.
    Composers outside TAG_VALUES get only an era; unknown ones get {}."""
    import re, unicodedata
    # strip accents so "Frédéric" matches "frederic", then lower-case
    plain = unicodedata.normalize("NFKD", name.split("/")[0])
    plain = "".join(c for c in plain if not unicodedata.combining(c)).lower()
    out = {}
    for era, names in ERA.items():
        for n in names:
            if re.search(rf"\b{n}\b", plain):
                out["era"] = era
                if n in TAG_VALUES["composer"]:
                    out["composer"] = n
                return out
    return out
