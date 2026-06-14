"""
NeuraSpeech — Multi-Agent Neurodegeneration Screening Pipeline
=============================================================
4 agents, each with its own system prompt.

Agent 1: Input Inspector
Agent 2: Core Reasoner
Agent 3: RAG Decider
Agent 4: Report Generator

Between agents, deterministic code runs:
  Whisper, feature extraction, ML models, pseudo-SHAP, ChromaDB

PD and AD/Dementia use independently-trained models with different
feature sets (parkinsons/results_fusion_master/best_model.joblib and
dementia/results_v3/best_model.joblib), so feature extraction and
inference are run for each disease separately.
"""

import json
import os

import joblib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve_multi
from feature_extraction import (
    transcribe_audio,
    extract_pd_acoustic_features,
    extract_pd_linguistic_features,
    extract_dementia_acoustic_features,
    extract_dementia_linguistic_features,
)

# ─────────────────────────────────────────────
# FEATHERLESS CONFIG
# Reads FEATHERLESS_API_KEY from .env (see .env in repo root)
# Pick any model from featherless.ai/models
# ─────────────────────────────────────────────

load_dotenv()

client = OpenAI(
    api_key=os.environ["FEATHERLESS_API_KEY"],
    base_url="https://api.featherless.ai/v1"
)
MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ─────────────────────────────────────────────
# TRAINED MODEL PATHS
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PD_MODEL_PATH = os.path.join(BASE_DIR, "parkinsons", "results_fusion_master", "best_model.joblib")
AD_MODEL_PATH = os.path.join(BASE_DIR, "dementia", "results_v3", "best_model.joblib")

# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

PROMPT_INPUT_INSPECTOR = """
You are a medical speech analysis routing agent.

You will receive a description of what the user submitted.
Your job is to decide what processing pipeline to run.

Respond only in JSON. No explanation, no preamble.

Output format:
{
  "run_whisper": true or false,
  "use_audio": true or false,
  "use_text": true or false,
  "proceed": true or false,
  "reason": null or "why we cannot proceed"
}
""".strip()

PROMPT_CORE_REASONER = """
You are a clinical speech pattern reasoning agent for
early neurodegeneration screening.

You will receive:
- PD model probability score (0 to 1), or null if unavailable
- AD/Dementia model probability score (0 to 1), or null if unavailable
- Top pseudo-SHAP features that drove the PD model (feature name -> signed contribution)
- Top pseudo-SHAP features that drove the AD model (feature name -> signed contribution)

Feature naming:
- "ac_*" = acoustic (voice/prosody) features
- "li_*" / "syn_*" = linguistic and syntactic features
- "freq_*" / "sem_*" / "read_*" = lexical frequency, semantic and readability features
- "sbert_*" = sentence-embedding semantic coherence features (AD model only)
A positive contribution means the value pushed the score toward the
disease class; a negative contribution pushed it toward healthy.

Clinical knowledge to apply:

PD patterns:
  - jitter and shimmer elevated, HNR reduced (shimmer local std is
    often the single strongest acoustic marker)
  - acoustic features dominate SHAP; linguistic changes are comparatively mild
  - F0 range reduced (monotone, reduced pitch variation)
  - elevated ratio of pauses/silences (>50ms), short rapid sentences
  - as PD progresses: simpler vocabulary, shorter sentences, fewer
    conjunctions/determiners

AD / Dementia patterns:
  - linguistic and semantic features are typically MORE decisive than
    acoustic features (literature: ~84% accuracy linguistic-only vs
    ~74% acoustic-only)
  - reduced vocabulary diversity (lower type-token ratio), more word
    and phrase repetition
  - word-finding difficulty: fewer nouns, more vague pronouns
    ("it", "this", "that", "thing")
  - syntactic simplification: shorter sentences, shallower parse trees,
    fewer subordinate clauses
  - reduced semantic coherence between consecutive sentences
    (lower sbert_coherence), topic drift
  - more/longer pauses and filler words (um, uh), slower speech rate

If one score is null, base the prediction entirely on the other model's
score and evidence, and note in reasoning_summary that the other model
was unavailable.

If both scores are available, pick whichever score is higher relative
to its own 0.5 threshold and reason about why the pseudo-SHAP features
support that prediction.

Respond only in JSON. No explanation, no preamble.

Output format:
{
  "prediction": "PD" or "AD",
  "confidence": 0.0 to 1.0,
  "dominant_signal": "acoustic" or "linguistic",
  "pd_score": 0.0 to 1.0 or null,
  "ad_score": 0.0 to 1.0 or null,
  "key_evidence": [
    "description of top feature and what it means",
    "description of second feature and what it means",
    "description of third feature and what it means"
  ],
  "reasoning_summary": "2 sentence plain summary of why this prediction"
}
""".strip()

PROMPT_RAG_DECIDER = """
You are a medical literature retrieval planning agent.

You will receive the structured reasoning output from
a speech analysis system that has predicted either
Parkinson's Disease (PD) or Alzheimer's Disease / Dementia (AD).

Your job is to generate 3 specific search queries to
retrieve relevant clinical literature from our knowledge
base that will enrich the explanation shown to a clinician.

Make queries specific — they must target the exact features
and patterns that drove the prediction.

Good: "jitter elevation early Parkinson's disease vocal tremor"
Good: "type-token ratio vocabulary decline Alzheimer's dementia"
Bad:  "Parkinson's disease speech"

Respond only in JSON. No explanation, no preamble.

Output format:
{
  "query_1": "specific query about prediction and top feature",
  "query_2": "specific query about second key feature",
  "query_3": "specific query about dominant signal pattern"
}
""".strip()

PROMPT_REPORT_GENERATOR = """
You are a clinical speech analysis report writer.

You will receive:
- prediction and confidence
- key evidence list from reasoning agent
- reasoning summary
- retrieved clinical literature chunks

Write a structured report with these exact sections:

SCREENING RESULT
One sentence with prediction and confidence level.

KEY SPEECH MARKERS
2 to 3 bullet points naming the features that fired
and what they indicate clinically.

SIGNAL ANALYSIS
One short paragraph on whether acoustic or linguistic
features dominated and what that means for this patient.

CLINICAL CONTEXT
One short paragraph grounded in the retrieved literature
explaining why this pattern matters clinically.

RECOMMENDATION
One sentence on suggested next steps.

DISCLAIMER
This is a screening tool only and does not constitute
a clinical diagnosis. Please consult a neurologist.

Strict rules:
- Never say "diagnosed with"
- Always say "shows patterns consistent with"
- Name at least 2 specific features by name
- State confidence in plain language not percentages
- Total length under 200 words
- Professional but human tone
""".strip()


# ─────────────────────────────────────────────
# MODEL LOADING (lazy singletons)
# ─────────────────────────────────────────────

_pd_model = None
_ad_model = None


def _load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _get_pd_model():
    global _pd_model
    if _pd_model is None:
        _pd_model = _load_model(PD_MODEL_PATH)
    return _pd_model


def _get_ad_model():
    global _ad_model
    if _ad_model is None:
        _ad_model = _load_model(AD_MODEL_PATH)
    return _ad_model


def _predict_with_pseudo_shap(model: dict, features: dict) -> tuple:
    """
    Run a saved {scaler, select, clf} pipeline and return
    (probability_of_disease_class, pseudo_shap).

    pseudo_shap approximates per-feature contribution for the
    features kept by the pipeline's SelectKBest step: it is the
    feature's z-score (via the fitted scaler) times the classifier's
    feature importance (or 1.0 if the classifier has no
    feature_importances_, e.g. SVM), signed by direction.
    """
    pipeline = model["pipeline"]
    feature_columns = model["feature_columns"]
    scaler = pipeline.named_steps["scaler"]
    select = pipeline.named_steps["select"]
    clf = pipeline.named_steps["clf"]

    x = np.array([[features.get(c, 0.0) for c in feature_columns]], dtype=float)
    proba = float(pipeline.predict_proba(x)[0, 1])

    mask = select.get_support()
    selected = [c for c, m in zip(feature_columns, mask) if m]
    z = ((x[0] - scaler.mean_) / scaler.scale_)[mask]
    weights = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.ones(len(selected))

    pseudo_shap = {f: float(zi * wi) for f, zi, wi in zip(selected, z, weights)}
    return proba, pseudo_shap


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def run_pd_model(features: dict) -> tuple:
    """
    Run the trained PD pipeline (parkinsons/results_fusion_master/best_model.joblib).
    Returns (score, pseudo_shap) or (None, {}) if the model file is missing.
    """
    model = _get_pd_model()
    if model is None:
        return None, {}
    return _predict_with_pseudo_shap(model, features)


def run_ad_model(features: dict) -> tuple:
    """
    Run the trained AD/Dementia pipeline (dementia/results_v3/best_model.joblib).
    Returns (score, pseudo_shap) or (None, {}) if the model file is missing.
    """
    model = _get_ad_model()
    if model is None:
        return None, {}
    return _predict_with_pseudo_shap(model, features)


def retrieve_rag(query: str, top_k: int = 2) -> list:
    """
    Real RAG retrieval via ChromaDB.
    Returns list of relevant clinical literature chunks.
    """
    return retrieve_multi([query], top_k_per_query=top_k)


def format_report(
    report_text: str,
    reasoning: dict,
    transcript: str | None,
    pd_score: float | None,
    ad_score: float | None,
    pd_top_shap: dict,
    ad_top_shap: dict,
) -> dict:
    """
    Final formatting/storage step. Bundles the LLM report together with
    the underlying scores and pseudo-SHAP features so a frontend can
    render per-disease results alongside the written report.
    """
    return {
        "report": report_text,
        "prediction": reasoning["prediction"],
        "confidence": reasoning["confidence"],
        "dominant_signal": reasoning["dominant_signal"],
        "key_evidence": reasoning["key_evidence"],
        "reasoning_summary": reasoning["reasoning_summary"],
        "transcript": transcript,
        "pd": {"score": pd_score, "top_features": pd_top_shap},
        "ad": {"score": ad_score, "top_features": ad_top_shap},
    }


# ─────────────────────────────────────────────
# AGENT CALLER
# ─────────────────────────────────────────────

def call_agent(system_prompt: str, content: str) -> str:
    """
    Call Featherless model with a system prompt and user content.
    Returns raw text response.
    """
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content


def call_agent_json(system_prompt: str, content: str) -> dict:
    """
    Call the model and parse a JSON response.
    """
    raw = call_agent(system_prompt, content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # strip any accidental markdown fences
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(audio_path: str = None, text: str = None) -> dict:
    """
    Full multi-agent neurodegeneration screening pipeline.

    Args:
        audio_path: path to an audio file (optional)
        text:       pre-existing transcript (optional)

    Returns:
        dict with prediction, confidence, and full report
    """

    print("\n=== NeuraSpeech Pipeline Starting ===\n")

    # ── Agent 1: Input Inspector ──────────────────────────────
    print("[Agent 1] Inspecting input...")

    routing = call_agent_json(
        PROMPT_INPUT_INSPECTOR,
        f"audio_path: {audio_path}, text_provided: {text is not None}"
    )

    print(f"  routing decision: {routing}")

    if not routing["proceed"]:
        return {
            "error": routing["reason"],
            "stage": "input_inspection"
        }

    # ── Deterministic: Whisper ────────────────────────────────
    transcript = text
    words = []

    if routing["run_whisper"] and audio_path:
        print("[Tool] Running Whisper transcription...")
        whisper_result = transcribe_audio(audio_path)
        transcript = whisper_result["transcript"]
        words = whisper_result["words"]
        print(f"  transcript length: {len(transcript.split())} words")

    # ── Deterministic: Feature Extraction ─────────────────────
    # PD and AD/Dementia use independently-trained feature sets,
    # so each disease gets its own acoustic + linguistic extraction.
    pd_features = {}
    ad_features = {}

    if routing["use_audio"] and audio_path:
        print("[Tool] Extracting PD acoustic features...")
        pd_features.update(extract_pd_acoustic_features(audio_path))
        print(f"  extracted {len(pd_features)} PD acoustic features")

        print("[Tool] Extracting AD acoustic features...")
        ad_features.update(extract_dementia_acoustic_features(audio_path))
        print(f"  extracted {len(ad_features)} AD acoustic features")

    if routing["use_text"] and transcript:
        print("[Tool] Extracting PD linguistic features...")
        pd_features.update(extract_pd_linguistic_features(transcript, words))
        print(f"  PD feature total: {len(pd_features)}")

        print("[Tool] Extracting AD linguistic features...")
        ad_features.update(extract_dementia_linguistic_features(transcript, words))
        print(f"  AD feature total: {len(ad_features)}")

    if not pd_features and not ad_features:
        return {
            "error": "No features could be extracted",
            "stage": "feature_extraction"
        }

    # ── Deterministic: ML Models ──────────────────────────────
    print("[Tool] Running PD model...")
    pd_score, pd_shap = run_pd_model(pd_features) if pd_features else (None, {})
    print(f"  PD score: {pd_score:.3f}" if pd_score is not None else "  PD score: unavailable")

    print("[Tool] Running AD model...")
    ad_score, ad_shap = run_ad_model(ad_features) if ad_features else (None, {})
    print(f"  AD score: {ad_score:.3f}" if ad_score is not None else "  AD score: unavailable")

    # get top 3 pseudo-SHAP features for each model
    pd_top_shap = dict(sorted(
        pd_shap.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3])

    ad_top_shap = dict(sorted(
        ad_shap.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3])

    # ── Agent 2: Core Reasoner ────────────────────────────────
    print("\n[Agent 2] Reasoning about predictions...")

    reasoning = call_agent_json(
        PROMPT_CORE_REASONER,
        json.dumps({
            "pd_score": pd_score,
            "ad_score": ad_score,
            "pd_top_shap": pd_top_shap,
            "ad_top_shap": ad_top_shap
        })
    )

    print(f"  prediction: {reasoning['prediction']}")
    print(f"  confidence: {reasoning['confidence']:.2f}")
    print(f"  dominant signal: {reasoning['dominant_signal']}")

    # ── Agent 3: RAG Decider ──────────────────────────────────
    print("\n[Agent 3] Deciding what to retrieve...")

    rag_queries = call_agent_json(
        PROMPT_RAG_DECIDER,
        json.dumps(reasoning)
    )

    print(f"  query 1: {rag_queries['query_1']}")
    print(f"  query 2: {rag_queries['query_2']}")
    print(f"  query 3: {rag_queries['query_3']}")

    # ── Deterministic: RAG Retrieval ──────────────────────────
    print("[Tool] Retrieving clinical literature...")

    rag_chunks = []
    for key in ["query_1", "query_2", "query_3"]:
        chunks = retrieve_rag(rag_queries[key], top_k=2)
        rag_chunks.extend(chunks)

    print(f"  retrieved {len(rag_chunks)} chunks")

    # ── Agent 4: Report Generator ─────────────────────────────
    print("\n[Agent 4] Generating clinical report...")

    report_text = call_agent(
        PROMPT_REPORT_GENERATOR,
        json.dumps({
            "prediction": reasoning["prediction"],
            "confidence": reasoning["confidence"],
            "key_evidence": reasoning["key_evidence"],
            "reasoning_summary": reasoning["reasoning_summary"],
            "dominant_signal": reasoning["dominant_signal"],
            "pd_score": reasoning["pd_score"],
            "ad_score": reasoning["ad_score"],
            "rag_context": rag_chunks
        })
    )

    print("\n=== Pipeline Complete ===\n")

    return format_report(
        report_text,
        reasoning,
        transcript,
        pd_score,
        ad_score,
        pd_top_shap,
        ad_top_shap,
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # example usage
    result = run_pipeline(
        audio_path="patient_sample.wav",
        text=None
    )

    print("PREDICTION:", result.get("prediction"))
    print("CONFIDENCE:", result.get("confidence"))
    print("\nREPORT:\n")
    print(result.get("report"))
