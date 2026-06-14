"""
Clean up redundant li_* columns and add spaCy-based syntactic/semantic
features (POS ratios, dependency-tree depth/distance, clause structure,
idea density, entity density) extracted from whisper_transcript.

Run once per master dataset:
    python3 add_syntactic_features.py dementia_master_dataset.csv
    python3 add_syntactic_features.py parkinsons_master_dataset.csv
"""

import sys
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

# Columns that are exact/near-exact duplicates of other li_ columns
# (see analysis: r >= 0.95, or definitionally redundant e.g. A = 1 - B)
DROP_COLS = [
    "li_exclamation_ratio",   # constant 0 in both datasets
    "li_func_word_ratio",     # = 1 - li_lexical_density (r=1.0)
    "li_content_word_ratio",  # = li_lexical_density (r=1.0)
    "li_filler_count",        # = li_filler_ratio * word_count (r=1.0)
    "li_vocab_size",          # r=0.95-0.97 with li_word_count; TTR already
                               # encodes vocab/word_count ratio
    "li_chars_per_sec",       # r=0.95-0.98 with li_speech_rate_wps
    "li_trigram_rep",         # r=0.97 with li_bigram_rep
]

SUBORDINATE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass"}
IDEA_POS = {"VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ"}

SYN_FEATURE_NAMES = [
    "syn_noun_ratio", "syn_verb_ratio", "syn_adj_ratio", "syn_adv_ratio",
    "syn_noun_verb_ratio", "syn_mean_dep_distance", "syn_mean_parse_depth",
    "syn_subordinate_clause_ratio", "syn_clauses_per_sentence",
    "syn_idea_density", "syn_entity_density",
]


def tree_depth(token, depth=0):
    children = list(token.children)
    if not children:
        return depth
    return max(tree_depth(c, depth + 1) for c in children)


def extract_syntactic_features(text):
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in SYN_FEATURE_NAMES}

    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    n_tokens = len(tokens)
    sents = list(doc.sents)
    n_sents = max(len(sents), 1)

    pos_counts = Counter(t.pos_ for t in tokens)
    noun = pos_counts.get("NOUN", 0)
    verb = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    adj = pos_counts.get("ADJ", 0)
    adv = pos_counts.get("ADV", 0)

    dep_distances = [abs(t.i - t.head.i) for t in doc if t.dep_ != "ROOT" and not t.is_space]
    mean_dep_dist = float(np.mean(dep_distances)) if dep_distances else 0.0

    depths = [tree_depth(sent.root) for sent in sents]
    mean_depth = float(np.mean(depths)) if depths else 0.0

    sub_clause_count = sum(1 for t in doc if t.dep_ in SUBORDINATE_DEPS)
    clause_count = sum(
        1 for t in doc
        if t.pos_ in ("VERB", "AUX") and (t.dep_ == "ROOT" or t.dep_ in SUBORDINATE_DEPS or t.dep_ == "conj")
    )

    idea_count = sum(1 for t in tokens if t.pos_ in IDEA_POS)
    n_ents = len(doc.ents)

    return {
        "syn_noun_ratio": noun / max(n_tokens, 1),
        "syn_verb_ratio": verb / max(n_tokens, 1),
        "syn_adj_ratio": adj / max(n_tokens, 1),
        "syn_adv_ratio": adv / max(n_tokens, 1),
        "syn_noun_verb_ratio": noun / max(verb, 1),
        "syn_mean_dep_distance": mean_dep_dist,
        "syn_mean_parse_depth": mean_depth,
        "syn_subordinate_clause_ratio": sub_clause_count / n_sents,
        "syn_clauses_per_sentence": clause_count / n_sents,
        "syn_idea_density": idea_count / max(n_tokens, 1) * 10,
        "syn_entity_density": n_ents / max(n_tokens, 1),
    }


def main(path):
    backup = path + ".bak"
    shutil.copy(path, backup)
    print(f"Backed up {path} -> {backup}")

    df = pd.read_csv(path)

    present = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=present)
    print(f"Dropped {len(present)} redundant columns: {present}")

    print(f"Extracting spaCy syntactic features for {len(df)} transcripts ...")
    records = []
    for i, text in enumerate(df["whisper_transcript"]):
        records.append(extract_syntactic_features(text))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}")

    syn_df = pd.DataFrame(records)
    df = pd.concat([df.reset_index(drop=True), syn_df], axis=1)

    df.to_csv(path, index=False)
    print(f"Saved {path}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"New columns: {SYN_FEATURE_NAMES}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
