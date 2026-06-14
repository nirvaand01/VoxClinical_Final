"""
Feature extraction backends for the NeuraSpeech agent pipeline.

A single per-clip feature set (163 features) feeds both the PD and
AD/Dementia models, plus a shared transcription helper:

  transcribe_audio                <- Whisper (base.en)
  extract_pd_acoustic_features    <- 115 ac_* librosa features
                                      (matches parkinsons_fusion_master.py
                                       / best_model.joblib, one row per clip)
  extract_pd_linguistic_features  <- li_*(30) + syn_*(11) + freq_/sem_/read_(7)
                                      = 48 features (matches the same model)

dementia_master_dataset.csv's 163 feature columns are exactly this same
ac_*/li_*/syn_*/freq_*/read_*/sem_* set, one row per clip (no subject-level
aggregation), which is what dementia/results_v3/best_model.joblib (v4
retrain) is trained on.

The PD li_*/syn_*/freq_/sem_/read_ set reconstructs the columns of
parkinsons_master_dataset.csv from model2_linguistic.py +
add_syntactic_features.py + add_nltk_features.py. A handful of columns
(li_max_word_len, li_pronoun_ratio, li_question_ratio, li_sent_len_cv,
li_word_dur_cv, li_rhythm_regularity) were not present verbatim in any
existing script and are defined here from their names using standard
formulas.
"""

import re
import math
from collections import Counter

import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# Lazy singletons — heavy models are loaded on first use only
# ─────────────────────────────────────────────────────────────────────────
_whisper_model = None
_nlp = None
_brown_freq = None
_total_brown = None
_top5000 = None
_stopwords = None
_cmu = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base.en")
    return _whisper_model


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_nltk_resources():
    global _brown_freq, _total_brown, _top5000, _stopwords, _cmu
    if _brown_freq is None:
        import nltk
        from nltk.corpus import brown, stopwords, cmudict
        _stopwords = set(stopwords.words("english"))
        _cmu = cmudict.dict()
        _brown_freq = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())
        _total_brown = sum(_brown_freq.values())
        _top5000 = {w for w, _ in _brown_freq.most_common(5000)}
    return _brown_freq, _total_brown, _top5000, _stopwords, _cmu


# ─────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION (Whisper base.en, Python API)
# ─────────────────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> dict:
    """
    Returns:
      {
        "transcript": "full text",
        "segments": [{"start": float, "end": float, "text": str}, ...],
        "words": [[word, start, end], ...],   # flat word-level timestamps
      }
    """
    model = _get_whisper()
    result = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        fp16=False,
        condition_on_previous_text=False,
        verbose=False,
    )

    text = result.get("text", "").strip()
    segments, words = [], []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"].strip(),
        })
        for w in seg.get("words", []):
            words.append([w["word"].strip(), w["start"], w["end"]])

    return {"transcript": text, "segments": segments, "words": words}


# ─────────────────────────────────────────────────────────────────────────
# SHARED ACOUSTIC EXTRACTION (librosa) — identical for PD and AD
# ─────────────────────────────────────────────────────────────────────────
def _extract_acoustic_raw(audio_path: str) -> dict:
    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    f = {}

    # MFCCs (13) + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1)), 1):
        f[f"ac_mfcc{i}_mean"] = float(m); f[f"ac_mfcc{i}_std"] = float(s)

    d1 = librosa.feature.delta(mfcc)
    for i, (m, s) in enumerate(zip(d1.mean(axis=1), d1.std(axis=1)), 1):
        f[f"ac_dmfcc{i}_mean"] = float(m); f[f"ac_dmfcc{i}_std"] = float(s)

    d2 = librosa.feature.delta(mfcc, order=2)
    for i, (m, s) in enumerate(zip(d2.mean(axis=1), d2.std(axis=1)), 1):
        f[f"ac_d2mfcc{i}_mean"] = float(m); f[f"ac_d2mfcc{i}_std"] = float(s)

    # F0 / voicing
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v = f0[voiced] if voiced.any() else np.array([0.0])
    f["ac_f0_mean"]         = float(np.mean(f0v))
    f["ac_f0_std"]          = float(np.std(f0v))
    f["ac_f0_range"]        = float(np.ptp(f0v))
    f["ac_f0_jitter"]       = float(np.mean(np.abs(np.diff(f0v)))) if len(f0v) > 1 else 0.0
    f["ac_voiced_fraction"] = float(voiced.mean())

    # RMS / shimmer
    rms = librosa.feature.rms(y=y)[0]
    f["ac_rms_mean"]       = float(rms.mean())
    f["ac_rms_std"]        = float(rms.std())
    f["ac_shimmer_approx"] = float(np.mean(np.abs(np.diff(rms))) / (rms.mean() + 1e-9))

    # Spectral
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    sct = librosa.feature.spectral_contrast(y=y, sr=sr)
    f["ac_spec_centroid_mean"]  = float(sc.mean());  f["ac_spec_centroid_std"]  = float(sc.std())
    f["ac_spec_bandwidth_mean"] = float(sb.mean());  f["ac_spec_bandwidth_std"] = float(sb.std())
    f["ac_spec_rolloff_mean"]   = float(sro.mean()); f["ac_spec_rolloff_std"]   = float(sro.std())
    f["ac_spec_flatness_mean"]  = float(sf.mean());  f["ac_spec_flatness_std"]  = float(sf.std())
    for b in range(sct.shape[0]):
        f[f"ac_spec_contrast_b{b+1}"] = float(sct[b].mean())

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f["ac_zcr_mean"] = float(zcr.mean()); f["ac_zcr_std"] = float(zcr.std())

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["ac_chroma_mean"] = float(chroma.mean()); f["ac_chroma_std"] = float(chroma.std())

    # Mel
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    f["ac_mel_mean"] = float(mel_db.mean()); f["ac_mel_std"] = float(mel_db.std())

    # Tonnetz
    y_h = librosa.effects.harmonic(y)
    ton = librosa.feature.tonnetz(y=y_h, sr=sr)
    for t in range(ton.shape[0]):
        f[f"ac_tonnetz{t+1}"] = float(ton[t].mean())

    # Onset rate & pause ratio
    onsets   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = librosa.get_duration(y=y, sr=sr)
    f["ac_onset_rate"] = float(len(onsets) / max(duration, 1e-3))
    rms2 = librosa.feature.rms(y=y, hop_length=512)[0]
    f["ac_pause_ratio"] = float((rms2 < rms2.max() * 0.02).mean())

    return f


def extract_pd_acoustic_features(audio_path: str) -> dict:
    """115 ac_* features for the PD fusion model (best_model.joblib)."""
    return _extract_acoustic_raw(audio_path)


# ─────────────────────────────────────────────────────────────────────────
# PD LINGUISTIC: li_*(30) + syn_*(11) + freq_/sem_/read_(7) = 48 features
# ─────────────────────────────────────────────────────────────────────────
FUNCTION_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","must","can","could",
    "and","but","or","nor","so","yet","for","at","by","in","on","to","up","as","of",
    "if","then","than","that","this","these","those","it","its","i","you","he","she",
    "we","they","me","him","her","us","them","my","your","his","our","their","which",
    "who","what","when","where","how","with","from","into","through","during","not",
    "no","nor","both","either","neither","each","every","all","any","some","such",
}
FILLERS = {"uh","um","er","ah","like","okay","well","right","so"}

SUBORDINATE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass"}
IDEA_POS = {"VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ"}


def _tokenize(text: str) -> list:
    return re.findall(r"\b[a-z']+\b", text.lower())


def _tree_depth(token, depth=0):
    children = list(token.children)
    if not children:
        return depth
    return max(_tree_depth(c, depth + 1) for c in children)


def _pd_timing_features(text: str, words: list) -> dict:
    f = {}
    n_chars = len(text.replace(" ", ""))

    if len(words) >= 2:
        durs       = [e - s for _, s, e in words]
        gaps       = [words[i + 1][1] - words[i][2] for i in range(len(words) - 1)]
        gaps_clean = [g for g in gaps if g >= 0]
        total_sp   = sum(durs)
        total_time = words[-1][2] - words[0][1]

        avg_word_dur = float(np.mean(durs))
        std_word_dur = float(np.std(durs))
        avg_gap      = float(np.mean(gaps_clean)) if gaps_clean else 0.0
        std_gap      = float(np.std(gaps_clean))  if gaps_clean else 0.0

        f["li_avg_word_dur"]     = avg_word_dur
        f["li_std_word_dur"]     = std_word_dur
        f["li_avg_gap"]          = avg_gap
        f["li_std_gap"]          = std_gap
        f["li_max_pause"]        = float(max(gaps_clean)) if gaps_clean else 0.0
        f["li_long_pause_count"] = sum(1 for g in gaps_clean if g > 0.5)
        f["li_long_pause_ratio"] = f["li_long_pause_count"] / max(len(gaps_clean), 1)
        f["li_speech_rate_wps"]      = len(words) / max(total_time, 1e-3)
        f["li_articulation_rate"]    = len(words) / max(total_sp, 1e-3)
        f["li_speaking_time_ratio"]  = total_sp / max(total_time, 1e-3)
        f["li_word_dur_cv"]      = std_word_dur / avg_word_dur if avg_word_dur > 0 else 0.0
        # Rhythm regularity: inverse of inter-word gap variability (1 = perfectly regular)
        f["li_rhythm_regularity"] = 1.0 / (1.0 + std_gap)

        # Noun-finding pause: gap before long (>=6 char) words — proxy for word-retrieval delay
        long_word_pre_gaps = []
        for i, (word, _, _) in enumerate(words[1:], 1):
            if len(word) >= 6:
                g = words[i][1] - words[i - 1][2]
                if g >= 0:
                    long_word_pre_gaps.append(g)
        f["li_noun_finding_pause"] = float(np.mean(long_word_pre_gaps)) if long_word_pre_gaps else 0.0
    else:
        for k in ["li_avg_word_dur", "li_std_word_dur", "li_avg_gap", "li_std_gap",
                  "li_max_pause", "li_long_pause_count", "li_long_pause_ratio",
                  "li_speech_rate_wps", "li_articulation_rate", "li_speaking_time_ratio",
                  "li_word_dur_cv", "li_noun_finding_pause"]:
            f[k] = 0.0
        f["li_rhythm_regularity"] = 1.0

    return f


def extract_pd_linguistic_features(transcript: str, words: list) -> dict:
    """li_*(30) + syn_*(11) + freq_/sem_/read_(7) = 48 features."""
    text = transcript or ""
    f = {}

    tokens    = _tokenize(text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    n_tokens  = len(tokens)
    n_sent    = max(len(sentences), 1)
    vocab     = set(tokens)

    # ── Lexical / sentence stats ───────────────────────────────────────────
    f["li_ttr"]        = len(vocab) / max(n_tokens, 1)
    f["li_word_count"] = n_tokens

    wl = [len(t) for t in tokens]
    f["li_avg_word_len"] = float(np.mean(wl)) if wl else 0.0
    f["li_std_word_len"] = float(np.std(wl))  if wl else 0.0
    f["li_max_word_len"] = float(max(wl))     if wl else 0.0

    sl = [len(_tokenize(s)) for s in sentences]
    avg_sent_len = float(np.mean(sl)) if sl else 0.0
    std_sent_len = float(np.std(sl))  if sl else 0.0
    f["li_avg_sent_len"]  = avg_sent_len
    f["li_std_sent_len"]  = std_sent_len
    f["li_num_sentences"] = n_sent
    f["li_sent_len_cv"]   = std_sent_len / avg_sent_len if avg_sent_len > 0 else 0.0

    # ── Lexical density (content words / total) ───────────────────────────
    n_func = sum(1 for t in tokens if t in FUNCTION_WORDS)
    f["li_lexical_density"] = 1.0 - (n_func / max(n_tokens, 1))

    # ── Repetition ──────────────────────────────────────────────────────────
    bigrams = list(zip(tokens, tokens[1:]))
    f["li_bigram_rep"] = (len(bigrams) - len(set(bigrams))) / max(len(bigrams), 1)
    f["li_top_word_dominance"] = (Counter(tokens).most_common(1)[0][1] / n_tokens) if tokens else 0.0

    # ── Disfluencies / punctuation ─────────────────────────────────────────
    n_fill = sum(1 for t in tokens if t in FILLERS)
    f["li_filler_ratio"] = n_fill / max(n_tokens, 1)
    f["li_punct_density"] = sum(1 for c in text if c in ".,;:!?") / max(len(text), 1)
    f["li_commas_per_sentence"] = text.count(",") / max(n_sent, 1)
    f["li_question_ratio"] = sum(1 for s in sentences if s.endswith("?")) / n_sent

    # ── spaCy: pronoun ratio + syntactic features ──────────────────────────
    nlp = _get_nlp()
    doc = nlp(text if text.strip() else " ")
    sp_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    n_sp = max(len(sp_tokens), 1)
    sents = list(doc.sents)
    n_sp_sent = max(len(sents), 1)

    pos_counts = Counter(t.pos_ for t in sp_tokens)
    noun = pos_counts.get("NOUN", 0)
    verb = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    adj  = pos_counts.get("ADJ", 0)
    adv  = pos_counts.get("ADV", 0)

    f["li_pronoun_ratio"] = pos_counts.get("PRON", 0) / n_sp

    dep_distances = [abs(t.i - t.head.i) for t in doc if t.dep_ != "ROOT" and not t.is_space]
    mean_dep_dist = float(np.mean(dep_distances)) if dep_distances else 0.0

    depths = [_tree_depth(sent.root) for sent in sents]
    mean_depth = float(np.mean(depths)) if depths else 0.0

    sub_clause_count = sum(1 for t in doc if t.dep_ in SUBORDINATE_DEPS)
    clause_count = sum(
        1 for t in doc
        if t.pos_ in ("VERB", "AUX") and (t.dep_ == "ROOT" or t.dep_ in SUBORDINATE_DEPS or t.dep_ == "conj")
    )
    idea_count = sum(1 for t in sp_tokens if t.pos_ in IDEA_POS)
    n_ents = len(doc.ents)

    f["syn_noun_ratio"] = noun / max(n_sp, 1)
    f["syn_verb_ratio"] = verb / max(n_sp, 1)
    f["syn_adj_ratio"]  = adj  / max(n_sp, 1)
    f["syn_adv_ratio"]  = adv  / max(n_sp, 1)
    f["syn_noun_verb_ratio"] = noun / max(verb, 1)
    f["syn_mean_dep_distance"] = mean_dep_dist
    f["syn_mean_parse_depth"]  = mean_depth
    f["syn_subordinate_clause_ratio"] = sub_clause_count / n_sp_sent
    f["syn_clauses_per_sentence"]     = clause_count / n_sp_sent
    f["syn_idea_density"]   = idea_count / max(n_sp, 1) * 10
    f["syn_entity_density"] = n_ents / max(n_sp, 1)

    # ── Timing features from Whisper word timestamps ───────────────────────
    f.update(_pd_timing_features(text, words))

    # ── NLTK: frequency / semantic / readability ───────────────────────────
    f.update(_extract_nltk_features(text))

    return f


def _extract_nltk_features(text: str) -> dict:
    from nltk import pos_tag
    from nltk.corpus import wordnet as wn

    names = ["freq_mean_log_freq", "freq_rare_word_ratio",
             "sem_avg_polysemy", "sem_avg_synset_depth",
             "read_avg_syllables_per_word", "read_flesch_kincaid_grade",
             "read_flesch_reading_ease"]

    tokens = _tokenize(text)
    if not tokens:
        return {k: 0.0 for k in names}

    brown_freq, total_brown, top5000, stop_words, cmu = _get_nltk_resources()

    sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    n_sentences = max(len(sentences), 1)
    content_words = [t for t in tokens if t not in stop_words]

    # lexical frequency / sophistication
    log_freqs, rare = [], 0
    for w in content_words:
        freq = brown_freq.get(w, 0)
        log_freqs.append(math.log10((freq + 1) / total_brown))
        if w not in top5000:
            rare += 1
    mean_log_freq = float(np.mean(log_freqs)) if log_freqs else 0.0
    rare_ratio = rare / max(len(content_words), 1)

    # WordNet semantic specificity
    tagged = pos_tag(tokens)
    polysemies, noun_depths = [], []
    for word, tag in tagged:
        if word in stop_words:
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

    # readability
    def count_syllables(word):
        if word in cmu:
            return sum(1 for ph in cmu[word][0] if ph[-1].isdigit())
        groups = re.findall(r"[aeiouy]+", word)
        return max(len(groups), 1)

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

