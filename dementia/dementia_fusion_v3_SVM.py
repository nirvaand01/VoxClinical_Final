"""
=============================================================
MODEL 7 — Fusion v3 (Acoustic + Linguistic + Semantic)
          Dementia Detection
Disease  : Dementia
Modality : Acoustic + Linguistic (spaCy) + SBERT Semantic (fusion)
Dataset  : 80% of subjects sampled per class
Transcript: Whisper base.en (word timestamps)
Features : Acoustic (115) + Linguistic/spaCy (~40) + SBERT (4)
           × mean+std+range aggregation per subject
           MI selection keeps top 30
Models   : XGBoost, Random Forest, SVM, Voting Ensemble
CV       : 5-Fold Stratified CV
Results  : Accuracy 0.778 | Precision 0.738 | Recall 0.672
           F1 0.703 | ROC-AUC 0.811  (best model: SVM)

Changes from v2:
  - Whisper base.en (kept for speed; spaCy layered on top for richer text features)
  - spaCy en_core_web_sm replaces regex parsing:
      real POS ratios, lemma-TTR, dependency distance, parse depth, NER
  - Removed redundant/noise-prone features:
      li_content_word_ratio (= 1-func_ratio), li_lexical_density (same),
      li_max_word_len (outlier-dominated), li_vocab_regression (hardcoded list),
      li_safe_word_ratio (12-word list), li_punct_density + li_commas_per_sentence
      (Whisper punctuation is unreliable)
  - Timing features retained from Whisper timestamps (complementary to spaCy)
=============================================================
"""

import os, re, json, warnings, random
import joblib
import numpy as np
import pandas as pd
import librosa
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.join(os.path.dirname(BASE), "dementia_data")
DEM_DIR   = os.path.join(DATA_BASE, "dementia")
NODEM_DIR = os.path.join(DATA_BASE, "nodementia")
OUT       = os.path.join(BASE, "results_v3")
os.makedirs(OUT, exist_ok=True)

ACOU_CACHE  = os.path.join(OUT, "acoustic_features.csv")
TRANS_CACHE = os.path.join(OUT, "transcripts.csv")
SBERT_CACHE = os.path.join(OUT, "sbert_features.csv")

print("Loading spaCy model …")
nlp = spacy.load("en_core_web_sm")


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
def sample_subjects(folder, label, pct=0.80, seed=42):
    subjects = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    n = max(1, int(len(subjects) * pct))
    random.seed(seed)
    chosen = set(random.sample(subjects, n))
    rows = []
    for subj in chosen:
        subj_path = os.path.join(folder, subj)
        for fname in sorted(os.listdir(subj_path)):
            if fname.lower().endswith(".wav"):
                rows.append({
                    "subject": subj,
                    "file": fname,
                    "path": os.path.join(subj_path, fname),
                    "label": label,
                })
    return rows

dem_rows   = sample_subjects(DEM_DIR,   1)
nodem_rows = sample_subjects(NODEM_DIR, 0)
manifest   = pd.DataFrame(dem_rows + nodem_rows)
print(f"Subjects: {manifest['subject'].nunique()} | Clips: {len(manifest)}")
print(f"  Dementia    : {manifest[manifest.label==1]['subject'].nunique()} subjects")
print(f"  No-Dementia : {manifest[manifest.label==0]['subject'].nunique()} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# ACOUSTIC FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_acoustic(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    f = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1)), 1):
        f[f"ac_mfcc{i}_mean"] = m;  f[f"ac_mfcc{i}_std"] = s
    d1 = librosa.feature.delta(mfcc)
    for i, (m, s) in enumerate(zip(d1.mean(axis=1), d1.std(axis=1)), 1):
        f[f"ac_dmfcc{i}_mean"] = m;  f[f"ac_dmfcc{i}_std"] = s
    d2 = librosa.feature.delta(mfcc, order=2)
    for i, (m, s) in enumerate(zip(d2.mean(axis=1), d2.std(axis=1)), 1):
        f[f"ac_d2mfcc{i}_mean"] = m;  f[f"ac_d2mfcc{i}_std"] = s

    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v = f0[voiced] if voiced.any() else np.array([0.0])
    f["ac_f0_mean"]         = float(np.mean(f0v))
    f["ac_f0_std"]          = float(np.std(f0v))
    f["ac_f0_range"]        = float(np.ptp(f0v))
    f["ac_f0_jitter"]       = float(np.mean(np.abs(np.diff(f0v)))) if len(f0v) > 1 else 0.0
    f["ac_voiced_fraction"] = float(voiced.mean())

    rms = librosa.feature.rms(y=y)[0]
    f["ac_rms_mean"]       = float(rms.mean())
    f["ac_rms_std"]        = float(rms.std())
    f["ac_shimmer_approx"] = float(np.mean(np.abs(np.diff(rms))) / (rms.mean() + 1e-9))

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

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f["ac_zcr_mean"] = float(zcr.mean());  f["ac_zcr_std"] = float(zcr.std())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["ac_chroma_mean"] = float(chroma.mean());  f["ac_chroma_std"] = float(chroma.std())

    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    f["ac_mel_mean"] = float(mel_db.mean());  f["ac_mel_std"] = float(mel_db.std())

    y_h = librosa.effects.harmonic(y)
    ton = librosa.feature.tonnetz(y=y_h, sr=sr)
    for t in range(ton.shape[0]):
        f[f"ac_tonnetz{t+1}"] = float(ton[t].mean())

    onsets   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = librosa.get_duration(y=y, sr=sr)
    f["ac_onset_rate"]  = float(len(onsets) / max(duration, 1e-3))
    rms2 = librosa.feature.rms(y=y, hop_length=512)[0]
    f["ac_pause_ratio"] = float((rms2 < rms2.max() * 0.02).mean())

    return f


if os.path.exists(ACOU_CACHE):
    print("\nLoading cached acoustic features …")
    acou_df = pd.read_csv(ACOU_CACHE)
else:
    print(f"\nExtracting acoustic features from {len(manifest)} clips …")
    records = []
    for i, row in manifest.iterrows():
        try:
            feat = extract_acoustic(row["path"])
            feat["subject"] = row["subject"]
            feat["file"]    = row["file"]
            feat["label"]   = row["label"]
            records.append(feat)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(manifest)} done …")
        except Exception as e:
            print(f"  SKIP {row['file']}: {e}")
    acou_df = pd.DataFrame(records)
    acou_df.to_csv(ACOU_CACHE, index=False)
    print(f"  Acoustic extraction complete: {len(acou_df)} clips")


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER base.en TRANSCRIPTION (Python API — no CLI dependency)
# ══════════════════════════════════════════════════════════════════════════════
import whisper as _whisper
print("Loading Whisper base.en model …")
_whisper_model = _whisper.load_model("base.en")

def transcribe(path):
    result = _whisper_model.transcribe(
        path,
        language="en",
        word_timestamps=True,
        fp16=False,
        condition_on_previous_text=False,
        verbose=False,
    )
    text  = result.get("text", "").strip()
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append([w["word"].strip(), w["start"], w["end"]])
    return {"text": text, "words": words}


if os.path.exists(TRANS_CACHE):
    print("\nLoading cached transcripts …")
    trans_df = pd.read_csv(TRANS_CACHE)
else:
    print(f"\nTranscribing {len(manifest)} clips with Whisper base.en …")
    rows = []
    for i, row in manifest.iterrows():
        print(f"  [{i+1}/{len(manifest)}] {row['subject']} / {row['file']}", flush=True)
        try:
            result = transcribe(row["path"])
            rows.append({
                "subject":    row["subject"],
                "file":       row["file"],
                "label":      row["label"],
                "text":       result["text"],
                "words_json": json.dumps(result["words"]),
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({
                "subject": row["subject"], "file": row["file"],
                "label": row["label"], "text": "", "words_json": "[]",
            })
    trans_df = pd.DataFrame(rows)
    trans_df.to_csv(TRANS_CACHE, index=False)
    print(f"  Transcription complete: {len(trans_df)} clips")


# ══════════════════════════════════════════════════════════════════════════════
# LINGUISTIC FEATURE EXTRACTION (spaCy + Whisper timestamps)
# ══════════════════════════════════════════════════════════════════════════════
FUNCTION_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","must","can","could",
    "and","but","or","nor","so","yet","for","at","by","in","on","to","up","as","of",
    "if","then","than","that","this","these","those","it","its","i","you","he","she",
    "we","they","me","him","her","us","them","my","your","his","our","their","which",
    "who","what","when","where","how","with","from","into","through","during","not",
    "no","nor","both","either","neither","each","every","all","any","some","such",
}
FILLERS   = {"uh","um","er","ah","hmm"}
# Vague reference pronouns — dementia patients use these instead of nouns
PRONOUNS  = {"he","she","it","they","him","her","them","this","that","these","those"}


def extract_linguistic(text, words_json):
    f = {}

    # ── spaCy parse ───────────────────────────────────────────────────────────
    doc    = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    n_tok  = max(len(tokens), 1)
    sents  = list(doc.sents)
    n_sent = max(len(sents), 1)

    # Basic lexical (spaCy tokens, not regex)
    words_lower = [t.text.lower() for t in tokens if t.is_alpha]
    n_words     = max(len(words_lower), 1)
    vocab       = set(words_lower)

    f["li_ttr"]         = len(vocab) / n_words
    f["li_vocab_size"]  = len(vocab)
    f["li_word_count"]  = n_words

    wl = [len(w) for w in words_lower]
    f["li_avg_word_len"] = np.mean(wl) if wl else 0.0
    f["li_std_word_len"] = np.std(wl)  if wl else 0.0

    # Lemma-TTR: proper vocabulary diversity after lemmatization
    content_lemmas = [t.lemma_.lower() for t in tokens if t.is_alpha and not t.is_stop]
    f["li_lemma_ttr"] = len(set(content_lemmas)) / max(len(content_lemmas), 1)

    # Sentence stats (spaCy sentence segmentation — better than regex split)
    sl = [len([t for t in s if t.is_alpha]) for s in sents]
    f["li_avg_sent_len"] = np.mean(sl) if sl else 0.0
    f["li_std_sent_len"] = np.std(sl)  if sl else 0.0
    f["li_num_sentences"]= n_sent

    # ── POS ratios (real tags, not hardcoded lists) ────────────────────────
    pos_counts = Counter(t.pos_ for t in tokens if t.is_alpha)
    f["li_noun_ratio"]  = pos_counts.get("NOUN", 0)  / n_words
    f["li_verb_ratio"]  = pos_counts.get("VERB", 0)  / n_words
    f["li_adj_ratio"]   = pos_counts.get("ADJ", 0)   / n_words
    f["li_adv_ratio"]   = pos_counts.get("ADV", 0)   / n_words
    f["li_pron_ratio"]  = pos_counts.get("PRON", 0)  / n_words
    # Noun-to-verb ratio: low in dementia (impoverished noun retrieval)
    f["li_noun_verb_ratio"] = pos_counts.get("NOUN", 0) / max(pos_counts.get("VERB", 1), 1)
    # Stop word ratio (replaces the crude function_word_ratio)
    n_stop = sum(1 for t in tokens if t.is_alpha and t.is_stop)
    f["li_stop_word_ratio"] = n_stop / n_words

    # ── Syntactic complexity (dependency parse) ────────────────────────────
    dep_dists = [abs(t.i - t.head.i) for t in tokens if t.dep_ != "ROOT"]
    f["li_mean_dep_dist"] = np.mean(dep_dists) if dep_dists else 0.0
    f["li_std_dep_dist"]  = np.std(dep_dists)  if dep_dists else 0.0
    # Parse tree depth per sentence
    def sent_depth(sent):
        root = [t for t in sent if t.dep_ == "ROOT"]
        if not root:
            return 0
        def depth(tok):
            children = list(tok.children)
            return 1 + max((depth(c) for c in children), default=0)
        return depth(root[0])
    depths = [sent_depth(s) for s in sents]
    f["li_avg_parse_depth"] = np.mean(depths) if depths else 0.0

    # ── NER (dementia patients produce fewer/wrong named entities) ─────────
    f["li_ner_count"]       = len(doc.ents)
    f["li_ner_per_sentence"]= len(doc.ents) / n_sent

    # ── Repetition / perseveration ─────────────────────────────────────────
    bigrams  = list(zip(words_lower, words_lower[1:]))
    trigrams = list(zip(words_lower, words_lower[1:], words_lower[2:]))
    f["li_bigram_rep"]   = (len(bigrams)  - len(set(bigrams)))  / max(len(bigrams),  1)
    f["li_trigram_rep"]  = (len(trigrams) - len(set(trigrams))) / max(len(trigrams), 1)
    f["li_top_word_dom"] = (Counter(words_lower).most_common(1)[0][1] / n_words) if words_lower else 0.0

    # ── Disfluencies ───────────────────────────────────────────────────────
    n_fill = sum(1 for w in words_lower if w in FILLERS)
    f["li_filler_ratio"] = n_fill / n_words
    f["li_filler_count"] = n_fill

    # ── Dementia-specific: vague pronoun overuse ───────────────────────────
    n_vague_pron = sum(1 for w in words_lower if w in PRONOUNS)
    f["li_vague_pronoun_ratio"] = n_vague_pron / n_words

    # ── Timing features (from Whisper base.en word timestamps) ────────────
    try:
        wl_list = json.loads(words_json)
    except Exception:
        wl_list = []

    if len(wl_list) >= 2:
        durs      = [e - s for _, s, e in wl_list]
        gaps      = [wl_list[i+1][1] - wl_list[i][2] for i in range(len(wl_list)-1)]
        gaps_c    = [g for g in gaps if g >= 0]
        total_sp  = sum(durs)
        total_time= wl_list[-1][2] - wl_list[0][1]

        f["li_avg_word_dur"]     = np.mean(durs)
        f["li_std_word_dur"]     = np.std(durs)
        f["li_avg_gap"]          = np.mean(gaps_c) if gaps_c else 0.0
        f["li_std_gap"]          = np.std(gaps_c)  if gaps_c else 0.0
        f["li_max_pause"]        = max(gaps_c)      if gaps_c else 0.0
        f["li_long_pause_count"] = sum(1 for g in gaps_c if g > 0.5)
        f["li_long_pause_ratio"] = f["li_long_pause_count"] / max(len(gaps_c), 1)
        f["li_speech_rate"]      = len(wl_list) / max(total_time, 1e-3)
        f["li_articu_rate"]      = len(wl_list) / max(total_sp,   1e-3)
        f["li_speaking_ratio"]   = total_sp / max(total_time, 1e-3)
        f["li_chars_per_sec"]    = len(text.replace(" ", "")) / max(total_time, 1e-3)

        # Noun-finding pause: gap before long words (word retrieval difficulty)
        long_word_pre_gaps = []
        for i, (word, _, _) in enumerate(wl_list[1:], 1):
            if len(word) >= 6:
                g = wl_list[i][1] - wl_list[i-1][2]
                if g >= 0:
                    long_word_pre_gaps.append(g)
        f["li_noun_finding_pause"] = np.mean(long_word_pre_gaps) if long_word_pre_gaps else 0.0
    else:
        for k in ["li_avg_word_dur","li_std_word_dur","li_avg_gap","li_std_gap",
                  "li_max_pause","li_long_pause_count","li_long_pause_ratio",
                  "li_speech_rate","li_articu_rate","li_speaking_ratio","li_chars_per_sec",
                  "li_noun_finding_pause"]:
            f[k] = 0.0

    return f


print("\nExtracting linguistic features (spaCy + Whisper timestamps) …")
ling_records = []
for i, row in trans_df.iterrows():
    feat = extract_linguistic(str(row["text"]), str(row["words_json"]))
    feat["subject"] = row["subject"]
    feat["file"]    = row["file"]
    ling_records.append(feat)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(trans_df)} done …")
ling_df = pd.DataFrame(ling_records)


# ══════════════════════════════════════════════════════════════════════════════
# SBERT SEMANTIC COHERENCE
# ══════════════════════════════════════════════════════════════════════════════
def extract_sbert(text, model):
    sents = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if len(s.strip()) > 10]
    if len(sents) < 2:
        return {"sbert_coherence_mean": 0.5, "sbert_coherence_std": 0.0,
                "sbert_coherence_min":  0.5, "sbert_topic_drift":   0.0}
    embs  = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    embs  = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sims  = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]
    return {
        "sbert_coherence_mean": np.mean(sims),
        "sbert_coherence_std":  np.std(sims),
        "sbert_coherence_min":  np.min(sims),
        "sbert_topic_drift":    float(np.dot(embs[0], embs[-1])),
    }

if os.path.exists(SBERT_CACHE):
    print("\nLoading cached SBERT features …")
    sbert_df = pd.read_csv(SBERT_CACHE)
else:
    print("\nComputing SBERT semantic coherence features …")
    sbert_model   = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_records = []
    for i, row in trans_df.iterrows():
        feat = extract_sbert(str(row["text"]), sbert_model)
        feat["subject"] = row["subject"]
        feat["file"]    = row["file"]
        sbert_records.append(feat)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(trans_df)} done …")
    sbert_df = pd.DataFrame(sbert_records)
    sbert_df.to_csv(SBERT_CACHE, index=False)
    print("  SBERT complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT-LEVEL AGGREGATION: mean + std + range
# ══════════════════════════════════════════════════════════════════════════════
acou_cols  = [c for c in acou_df.columns  if c.startswith("ac_")]
ling_cols  = [c for c in ling_df.columns  if c.startswith("li_")]
sbert_cols = [c for c in sbert_df.columns if c.startswith("sbert_")]

subject_labels = manifest.groupby("subject")["label"].first().reset_index()

def aggregate_subject(df, feat_cols, labels):
    rows = []
    for subj, grp in df.groupby("subject"):
        row = {"subject": subj}
        for col in feat_cols:
            vals = grp[col].dropna().values
            row[f"{col}_mean"]  = np.mean(vals) if len(vals) else 0.0
            row[f"{col}_std"]   = np.std(vals)  if len(vals) else 0.0
            row[f"{col}_range"] = (np.max(vals) - np.min(vals)) if len(vals) else 0.0
        rows.append(row)
    agg = pd.DataFrame(rows).merge(labels, on="subject")
    return agg

print("\nAggregating features per subject (mean + std + range) …")
acou_sub  = aggregate_subject(acou_df,  acou_cols,  subject_labels)
ling_sub  = aggregate_subject(ling_df,  ling_cols,  subject_labels)
sbert_sub = aggregate_subject(sbert_df, sbert_cols, subject_labels)

merged = acou_sub.merge(ling_sub.drop(columns="label"),   on="subject")
merged = merged.merge(sbert_sub.drop(columns="label"),    on="subject")

all_feat_cols = [c for c in merged.columns if c not in ("subject", "label")]
feat_modality = {}
for c in all_feat_cols:
    if c.startswith("ac_"):      feat_modality[c] = "Acoustic"
    elif c.startswith("li_"):    feat_modality[c] = "Linguistic"
    elif c.startswith("sbert_"): feat_modality[c] = "Semantic (SBERT)"

X        = merged[all_feat_cols].values.astype(float)
y_labels = merged["label"].values

n_dem   = y_labels.sum()
n_nodem = (y_labels == 0).sum()
spw     = n_nodem / n_dem

print(f"\nFinal dataset: {len(merged)} subjects")
print(f"  Dementia    : {n_dem}")
print(f"  No-Dementia : {n_nodem}")
print(f"  Total features before selection: {len(all_feat_cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON: XGBoost vs SVM vs Random Forest vs Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════
N_FEATURES = 30
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipeline(estimator):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=N_FEATURES)),
        ("smote",  SMOTE(random_state=42, k_neighbors=5,
                         sampling_strategy=min(1.0, n_nodem / n_dem * 1.5))),
        ("clf",    estimator),
    ])

models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    ),
}

print("\n" + "="*65)
print("  MODEL COMPARISON (5-Fold Stratified CV)")
print("="*65)

comparison_rows = []
best_auc   = 0
best_name  = None
best_pipe  = None

for name, estimator in models.items():
    pipe = make_pipeline(estimator)
    res  = cross_validate(pipe, X, y_labels, cv=cv,
                          scoring=["accuracy","precision","recall","f1","roc_auc"],
                          return_train_score=False)
    row = {
        "Model":     name,
        "Accuracy":  res["test_accuracy"].mean(),
        "Precision": res["test_precision"].mean(),
        "Recall":    res["test_recall"].mean(),
        "F1":        res["test_f1"].mean(),
        "ROC-AUC":   res["test_roc_auc"].mean(),
        "Acc_std":   res["test_accuracy"].std(),
        "AUC_std":   res["test_roc_auc"].std(),
    }
    comparison_rows.append(row)
    print(f"\n  {name}:")
    print(f"    Accuracy : {row['Accuracy']:.4f} ± {row['Acc_std']:.4f}")
    print(f"    Precision: {row['Precision']:.4f}")
    print(f"    Recall   : {row['Recall']:.4f}")
    print(f"    F1       : {row['F1']:.4f}")
    print(f"    ROC-AUC  : {row['ROC-AUC']:.4f} ± {row['AUC_std']:.4f}")

    if row["ROC-AUC"] > best_auc:
        best_auc  = row["ROC-AUC"]
        best_name = name
        best_pipe = make_pipeline(estimator)

# Voting Ensemble
print("\n  Building Voting Ensemble …")
ens_pipe = make_pipeline(VotingClassifier(estimators=[
    ("xgb", xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("rf",  RandomForestClassifier(
        n_estimators=400, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1)),
    ("svm", SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42)),
], voting="soft"))

res = cross_validate(ens_pipe, X, y_labels, cv=cv,
                     scoring=["accuracy","precision","recall","f1","roc_auc"],
                     return_train_score=False)
ens_row = {
    "Model":     "Voting Ensemble",
    "Accuracy":  res["test_accuracy"].mean(),
    "Precision": res["test_precision"].mean(),
    "Recall":    res["test_recall"].mean(),
    "F1":        res["test_f1"].mean(),
    "ROC-AUC":   res["test_roc_auc"].mean(),
    "Acc_std":   res["test_accuracy"].std(),
    "AUC_std":   res["test_roc_auc"].std(),
}
comparison_rows.append(ens_row)
print(f"\n  Voting Ensemble:")
print(f"    Accuracy : {ens_row['Accuracy']:.4f} ± {ens_row['Acc_std']:.4f}")
print(f"    Precision: {ens_row['Precision']:.4f}")
print(f"    Recall   : {ens_row['Recall']:.4f}")
print(f"    F1       : {ens_row['F1']:.4f}")
print(f"    ROC-AUC  : {ens_row['ROC-AUC']:.4f} ± {ens_row['AUC_std']:.4f}")

if ens_row["ROC-AUC"] > best_auc:
    best_auc  = ens_row["ROC-AUC"]
    best_name = "Voting Ensemble"
    best_pipe = ens_pipe

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print(f"\n  ★ Best model: {best_name} (ROC-AUC={best_auc:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING ON BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
if best_name == "XGBoost":
    print("\nHyperparameter tuning (XGBoost GridSearch) …")
    gs = GridSearchCV(best_pipe, {
        "clf__n_estimators":  [200, 400],
        "clf__max_depth":     [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
    }, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X, y_labels)
    best_pipe = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")
elif best_name == "Random Forest":
    print("\nHyperparameter tuning (RF GridSearch) …")
    gs = GridSearchCV(best_pipe, {
        "clf__n_estimators":     [200, 400, 600],
        "clf__max_depth":        [4, 6, 8, None],
        "clf__min_samples_leaf": [1, 2, 4],
    }, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X, y_labels)
    best_pipe = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")
else:
    print(f"\nSkipping GridSearch for {best_name}.")
    best_pipe.fit(X, y_labels)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nFinal CV evaluation with tuned {best_name} …")
y_pred_cv  = cross_val_predict(best_pipe, X, y_labels, cv=cv, method="predict")
y_proba_cv = cross_val_predict(best_pipe, X, y_labels, cv=cv, method="predict_proba")[:, 1]

acc  = accuracy_score(y_labels,  y_pred_cv)
prec = precision_score(y_labels, y_pred_cv, zero_division=0)
rec  = recall_score(y_labels,    y_pred_cv, zero_division=0)
f1   = f1_score(y_labels,        y_pred_cv, zero_division=0)
auc  = roc_auc_score(y_labels,   y_proba_cv)

print("\n── Final Results ──")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print()
print(classification_report(y_labels, y_pred_cv, target_names=["No Dementia","Dementia"]))

pd.DataFrame([{"Metric": m, "Value": v} for m, v in [
    ("Accuracy",acc),("Precision",prec),("Recall",rec),("F1",f1),("ROC-AUC",auc)
]]).to_csv(os.path.join(OUT, "final_metrics.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
comp_df.set_index("Model")[["Accuracy","Precision","Recall","F1","ROC-AUC"]].plot(
    kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)
ax.set_title("Dementia v3 — Model Comparison (5-Fold CV)")
ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_labels, y_pred_cv), display_labels=["No Dementia","Dementia"]
).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Best Model ({best_name}) — Confusion Matrix")
RocCurveDisplay.from_predictions(y_labels, y_proba_cv, ax=axes[1], name=best_name)
axes[1].set_title(f"ROC Curve (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE + SHAP
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting final model on full data for SHAP …")
best_pipe.fit(X, y_labels)

# ── Save the trained model (drop SMOTE: no-op at inference time) ────────
inference_pipe = Pipeline([
    ("scaler", best_pipe.named_steps["scaler"]),
    ("select", best_pipe.named_steps["select"]),
    ("clf",    best_pipe.named_steps["clf"]),
])
model_path = os.path.join(OUT, "best_model.joblib")
joblib.dump({
    "pipeline":        inference_pipe,
    "model_name":      best_name,
    "feature_columns": all_feat_cols,
    "n_features":      N_FEATURES,
    "cv_metrics":      comp_df[comp_df["Model"] == best_name].iloc[0].to_dict(),
}, model_path)
print(f"Saved trained pipeline -> {model_path}")

try:
    if best_name in ["XGBoost", "Voting Ensemble"]:
        raise ValueError("use XGBoost sub-model for SHAP")
    scaler_f = best_pipe.named_steps["scaler"]
    select_f = best_pipe.named_steps["select"]
    clf_f    = best_pipe.named_steps["clf"]
    sel_feat = [f for f, m in zip(all_feat_cols, select_f.get_support()) if m]
    X_sel    = select_f.transform(scaler_f.transform(X))
    imp_df   = pd.DataFrame({
        "feature":    sel_feat,
        "importance": clf_f.feature_importances_,
        "modality":   [feat_modality[f] for f in sel_feat],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
except Exception:
    print("  Using XGBoost sub-estimator for SHAP …")
    xgb_pipe = make_pipeline(xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    ))
    xgb_pipe.fit(X, y_labels)
    scaler_f = xgb_pipe.named_steps["scaler"]
    select_f = xgb_pipe.named_steps["select"]
    clf_f    = xgb_pipe.named_steps["clf"]
    sel_feat = [f for f, m in zip(all_feat_cols, select_f.get_support()) if m]
    X_sel    = select_f.transform(scaler_f.transform(X))
    imp_df   = pd.DataFrame({
        "feature":    sel_feat,
        "importance": clf_f.feature_importances_,
        "modality":   [feat_modality.get(f, "Unknown") for f in sel_feat],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
print(f"\nTop 15 features:\n{imp_df.head(15).to_string(index=False)}")

modal_imp = imp_df.groupby("modality")["importance"].sum().sort_values(ascending=False)
print(f"\nImportance by modality:\n{modal_imp.to_string()}")

X_df      = pd.DataFrame(X_sel, columns=sel_feat)
explainer = shap.TreeExplainer(clf_f)
shap_vals = explainer.shap_values(X_df)

palette = {"Acoustic": "#1565C0", "Linguistic": "#E64A19", "Semantic (SBERT)": "#2E7D32"}

plt.figure(figsize=(12, 9))
shap.summary_plot(shap_vals, X_df, show=False, max_display=25)
plt.title("SHAP Summary (Beeswarm) — Dementia v3")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=25)
plt.title("SHAP Feature Importance — Dementia v3")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(13, 9))
sns.barplot(data=imp_df, y="feature", x="importance",
            hue="modality", dodge=False, palette=palette, ax=ax)
ax.set_title("Feature Importances by Modality — Dementia v3")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(modal_imp.values, labels=modal_imp.index, autopct="%1.1f%%",
       colors=[palette[m] for m in modal_imp.index], startangle=90)
ax.set_title("Feature Importance by Modality — Dementia v3")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "modality_pie.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY vs v2
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FINAL SUMMARY — v2 (SVM, base.en) vs v3 (" + best_name + ", base.en + spaCy)")
print("="*65)
v2 = [("Accuracy","0.748"),("Precision","0.789"),("Recall","0.836"),("F1","0.812"),("ROC-AUC","0.692")]
v3 = [acc, prec, rec, f1, auc]
print(f"  {'Metric':<12} {'v2 (SVM)':>14} {'v3 ('+best_name+')':>20}")
print("-"*65)
for (name, v2val), v3val in zip(v2, v3):
    delta = v3val - float(v2val)
    arrow = "▲" if delta > 0 else "▼"
    print(f"  {name:<12} {v2val:>14} {v3val:>16.4f}  {arrow}{abs(delta):.4f}")
print("="*65)
print(f"\nAll results saved to: {OUT}")
print("Done.")
