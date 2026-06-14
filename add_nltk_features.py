"""
Add NLTK-based features extracted from whisper_transcript:
  - lexical frequency/sophistication (Brown corpus frequencies)
  - WordNet semantic specificity (polysemy, hypernym depth)
  - readability (syllables-per-word, Flesch-Kincaid)

Run once per master dataset:
    python3 add_nltk_features.py dementia_master_dataset.csv
    python3 add_nltk_features.py parkinsons_master_dataset.csv
"""

import re
import sys
import shutil
import math

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import brown, wordnet as wn, cmudict, stopwords
from nltk import pos_tag

STOPWORDS = set(stopwords.words("english"))
CMU = cmudict.dict()

# Brown-corpus frequency table (lowercase alpha words only)
_brown_freq = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())
_TOTAL_BROWN = sum(_brown_freq.values())
_TOP5000 = {w for w, _ in _brown_freq.most_common(5000)}

NLTK_FEATURE_NAMES = [
    "freq_mean_log_freq", "freq_rare_word_ratio",
    "sem_avg_polysemy", "sem_avg_synset_depth",
    "read_avg_syllables_per_word", "read_flesch_kincaid_grade", "read_flesch_reading_ease",
]


def tokenize(text):
    return re.findall(r"\b[a-z']+\b", text.lower())


def count_syllables(word):
    if word in CMU:
        return sum(1 for ph in CMU[word][0] if ph[-1].isdigit())
    # fallback heuristic: count vowel groups
    groups = re.findall(r"[aeiouy]+", word)
    return max(len(groups), 1)


def extract_nltk_features(text):
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in NLTK_FEATURE_NAMES}

    tokens = tokenize(text)
    if not tokens:
        return {k: 0.0 for k in NLTK_FEATURE_NAMES}

    sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    n_sentences = max(len(sentences), 1)

    content_words = [t for t in tokens if t not in STOPWORDS]

    # --- lexical frequency / sophistication ---
    log_freqs = []
    rare = 0
    for w in content_words:
        freq = _brown_freq.get(w, 0)
        log_freqs.append(math.log10((freq + 1) / _TOTAL_BROWN))
        if w not in _TOP5000:
            rare += 1
    mean_log_freq = float(np.mean(log_freqs)) if log_freqs else 0.0
    rare_ratio = rare / max(len(content_words), 1)

    # --- WordNet semantic specificity ---
    tagged = pos_tag(tokens)
    polysemies = []
    noun_depths = []
    for word, tag in tagged:
        if word in STOPWORDS:
            continue
        synsets = wn.synsets(word)
        if not synsets:
            continue
        polysemies.append(len(synsets))
        if tag.startswith("NN"):
            noun_synsets = wn.synsets(word, pos=wn.NOUN)
            if noun_synsets:
                noun_depths.append(noun_synsets[0].min_depth())
    avg_polysemy = float(np.mean(polysemies)) if polysemies else 0.0
    avg_synset_depth = float(np.mean(noun_depths)) if noun_depths else 0.0

    # --- readability ---
    syllables = [count_syllables(t) for t in tokens]
    avg_syllables = float(np.mean(syllables))
    words_per_sent = len(tokens) / n_sentences
    fk_grade = 0.39 * words_per_sent + 11.8 * avg_syllables - 15.59
    fre = 206.835 - 1.015 * words_per_sent - 84.6 * avg_syllables

    return {
        "freq_mean_log_freq": mean_log_freq,
        "freq_rare_word_ratio": rare_ratio,
        "sem_avg_polysemy": avg_polysemy,
        "sem_avg_synset_depth": avg_synset_depth,
        "read_avg_syllables_per_word": avg_syllables,
        "read_flesch_kincaid_grade": fk_grade,
        "read_flesch_reading_ease": fre,
    }


def main(path):
    backup = path + ".bak2"
    shutil.copy(path, backup)
    print(f"Backed up {path} -> {backup}")

    df = pd.read_csv(path)

    print(f"Extracting NLTK features for {len(df)} transcripts ...")
    records = []
    for i, text in enumerate(df["whisper_transcript"]):
        records.append(extract_nltk_features(text))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}")

    nltk_df = pd.DataFrame(records)
    df = pd.concat([df.reset_index(drop=True), nltk_df], axis=1)

    df.to_csv(path, index=False)
    print(f"Saved {path}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"New columns: {NLTK_FEATURE_NAMES}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
