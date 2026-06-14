"""
NeuraSpeech — RAG Module
========================
ChromaDB knowledge base built from two real papers:

PD:  Dudek et al. (2025) Sensors
     "Analysis of Voice, Speech, and Language Biomarkers
      of Parkinson's Disease Collected in a Mixed Reality Setting"

AD:  Qi et al. (2023) Frontiers in Aging Neuroscience
     "Noninvasive automatic detection of Alzheimer's disease
      from spontaneous speech: a review"

Usage:
    from rag import build_knowledge_base, retrieve, retrieve_multi

    build_knowledge_base()   # run once
    chunks = retrieve("jitter shimmer Parkinson's", top_k=2)
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ─────────────────────────────────────────────
# PD CHUNKS
# Source: Dudek et al. (2025) Sensors 25, 2405
# ─────────────────────────────────────────────

PD_CHUNKS = [

    """[PD | Acoustic Overview | Dudek et al. 2025]
    Parkinson's disease results from the deterioration of dopamine-producing neurons,
    causing tremors, muscle rigidity, bradykinesia, and issues with speech.
    Significant vocal tract changes are observable including dysphonia, monotony,
    reduced speech clarity, and increased frequency of speech interruptions.
    Parkinson's patients often exhibit weakened or incomplete vocal fold closure,
    leading to modulation difficulties. These changes stem from motor control
    disruptions affecting the laryngeal muscles and vocal tract structures.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Disease Progression Speech | Dudek et al. 2025]
    In Parkinson's disease, voice disorders typically originate with subtle changes
    in phonation and articulation but progressively worsen over time. Initially,
    slight pitch instability, significantly quieter speech with mild reductions in
    energy within the higher harmonic spectrum, monotonicity, and slowness may
    appear. As the condition advances, these alterations become more pronounced,
    leading to significant pitch fluctuations, a marked weakening of high-frequency
    components, and increasing imprecision in vowel and consonant articulation.
    Linguistically, Parkinson's patients may struggle with appropriate vocabulary
    selection, sentence complexity, and eloquence, tending to use more general
    words and shorter sentences over the disease progression.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Acoustic Features | Dudek et al. 2025]
    Key acoustic low-level descriptors for PD detection include: fundamental
    frequency F0 and its probability of voicing; jitter local and jitter delta
    measuring frequency perturbation; shimmer local measuring amplitude
    perturbation; RASTA-style filtered auditory spectral bands; spectral flux,
    entropy, variance, skewness, kurtosis, slope, and roll-off; spectral energy
    in 25-650 Hz and 1-4 kHz bands; mel-frequency cepstral coefficients MFCCs;
    RMS energy and zero crossing rate. Formants F1-F5 extracted via Parselmouth
    provide additional vocal tract shape information particularly for vowel tasks.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Silence and Pauses | Dudek et al. 2025]
    For spontaneous speech tasks such as picture descriptions, story retelling,
    and answering questions, the number and length of silences longer than 50ms
    provide valuable information about a patient's health condition. Silence
    intervals greater than 50ms can be classified as a pause in speech among
    people with Parkinson's disease. PD patients show a higher silences_number_ratio
    compared to healthy controls. Their speech is often unclear, very quiet, and
    consists of short, rapidly spoken sentences. The silence_number_ratio was among
    the top features in XGBoost models for story retelling tasks.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Shimmer as Top Feature | Dudek et al. 2025]
    Shimmer local standard deviation was the top SHAP feature for PD detection
    in both the diadochokinetic task and the story retelling task. In the
    diadochokinetic pataka task, shimmerLocal_sma_std showed a very large effect
    size (Cohen's d = 1.58, p < 0.0001). In the story retelling acoustic model,
    shimmerLocal_sma_std was the highest importance feature according to SHAP
    values. RASTA-filtered auditory spectral band features (audSpec_Rfilt) were
    also consistently among the most important features across tasks.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Linguistic Features | Dudek et al. 2025]
    Linguistic features extracted using spaCy from spontaneous speech tasks include:
    parts of speech POS tags including adverb, adjective, noun, verb, pronoun,
    conjunction, determiner; named entity recognition for people, dates, locations;
    word frequency analysis; normalized number of different words (type-token ratio);
    and sentiment analysis. In XGBoost models combining acoustic and linguistic
    features for story retelling, two linguistic features were among the top ten:
    CCONJ_POS (coordinating conjunctions) and DET_POS (determiners), both showing
    large effect sizes. Linguistic features showed moderate predictive power but
    were less decisive than acoustic features for PD.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | XGBoost Performance | Dudek et al. 2025]
    The XGBoost model achieved the best performance among all classifiers including
    logistic regression, SVM, random forests, and AdaBoost. For the story retelling
    task combining acoustic and linguistic features, XGBoost achieved F1-score of
    0.90 plus or minus 0.05, recall of 0.91 plus or minus 0.11, precision of 0.92
    plus or minus 0.10. For acoustic features only in story retelling, recall was
    0.95 plus or minus 0.01. The diadochokinetic pataka task achieved precision
    of 0.96 plus or minus 0.08. Feature selection of top 10 features based on
    importance and consistency across cross-validation folds significantly improved
    performance.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Cognitive Correlation MoCA | Dudek et al. 2025]
    Fundamental frequency F0 features showed strong positive correlations with
    Montreal Cognitive Assessment MoCA scores in PD patients. The feature
    pakata_F0final_sma_quantile50 showed r=0.7369 correlation with MoCA.
    More stable and well-modulated voicing as reflected by F0 measures is
    associated with better cognitive performance. F0 features from diadochokinetic
    tasks showed stronger associations with cognitive impairment, while vowel
    phonation tasks were more reflective of motor dysfunction as measured by
    MDS-UPDRS-III. This demonstrates that acoustic features can serve as proxies
    for both motor and cognitive assessment in PD.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Spontaneous vs Controlled Tasks | Dudek et al. 2025]
    Spontaneous monologue tasks such as story retelling provided the clearest
    differentiation between PD and healthy control groups. In these tasks, MFCCs,
    PCM FFT magnitudes, and shimmer emerged as major contributors to detecting
    typical PD-associated traits such as reduced pitch variation, breathiness, and
    vocal fatigue. The ratio and duration of silence segments stood out as influential
    markers, especially for identifying slowed articulation or festinating speech
    patterns. Story retelling combining acoustic and linguistic features outperformed
    picture description and daily activity description tasks.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

    """[PD | Acoustic vs Linguistic Signal | Dudek et al. 2025]
    Results showed that models trained only on linguistic features were significantly
    worse compared to those trained exclusively on acoustic features and never
    exceeded the values of models trained on acoustic features. The most influential
    acoustic features across tasks were MFCC, audSpec_Rfilt, pcm_fftMag, and
    shimmer. However, the collective evidence underlines that multifaceted vocal
    biomarkers provide a richer representation of neurological status, capturing
    both how patients speak through acoustics and what they say through linguistics.
    Source: Dudek et al. (2025) Sensors 25, 2405.""",

]

# ─────────────────────────────────────────────
# AD CHUNKS
# Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723
# ─────────────────────────────────────────────

AD_CHUNKS = [

    """[AD | Overview | Qi et al. 2023]
    Alzheimer's disease is one of the most prevalent neurological disorders,
    primarily affecting older adults and one of the main causes of death among
    people over 70. Dementia currently affects over 50 million people worldwide,
    likely increasing to 152 million in 2050. AD is characterized by continuous
    deterioration of cognitive and functional abilities including language, memory,
    attention and executive function. Speech-based methods offer non-invasive,
    convenient, and cost-effective solutions for automatic AD detection.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Why Speech is Useful | Qi et al. 2023]
    Speech is closely related to cognitive status and widely used in mental health
    assessment. The most significant correlations with AD are differences in speech
    comprehension, reasoning, language production, and memory, which result in
    reduction in vocabulary and verbal fluency, as well as difficulties in performing
    daily tasks related to semantic information. AD patients exhibit lower speech rate,
    more frequent and longer hesitations, obscurer pronunciation, and longer pauses
    compared to healthy controls. The hesitation ratio shows particularly notable
    disparities and temporal aspects of speech play a vital role in differentiating AD.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Acoustic Features | Qi et al. 2023]
    Acoustic features used in AD detection include frame-level features such as
    MFCCs capturing spectral information; jitter and shimmer assessing perturbations
    in fundamental frequency and amplitude; harmonics-to-noise ratio HNR providing
    measures of vocal stability; speech rate and pauses reflecting speech fluency;
    prosodic measures including temporal aspects, intensity, voice quality, and
    variation in F0; and disfluency features such as percentage of broken words,
    repetitions, sound prolongations, self-repairs, and pause duration. AD can impact
    coordination of muscles involved in speech, resulting in changes in articulation
    and reduced vocal range detectable through MFCCs and perturbation measures.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Pauses and Disfluency | Qi et al. 2023]
    AD patients often exhibit slower speech rate, longer pauses or breaks between
    words or sentences, and increased difficulty in finding the right words, resulting
    in disfluencies. AD patients had more and longer unfilled pauses. Filled pauses
    including um, uh, oh, well, laughter are common. AD patients showed potential to
    use more uh and laughter and meaningless words like well and oh, but less um
    compared to healthy controls. Encoding pauses into long over 2 seconds, medium
    0.5-2 seconds and short under 0.5 seconds and combining with language models
    achieved 89.58% accuracy.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Linguistic Features Overview | Qi et al. 2023]
    Linguistic features undergo changes in AD due to progressive impairment
    affecting word retrieval, comprehension of complex grammatical structures,
    construction of grammatically correct sentences, and maintenance of coherent
    discourse. Language impairments are evident in alterations in vocabulary usage,
    sentence structure, and overall linguistic fluency. AD can result in decreased
    verbal expression including reduced output, shorter and less complex sentences,
    and decrease in overall quantity of speech. The range of vocabulary becomes
    limited and utilization of syntactic structures may diminish.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | POS and Lexical Features | Qi et al. 2023]
    Parts-of-speech POS reflect language changes in AD including a decrease in
    number of nouns and increase in number of pronouns, adjectives and verbs.
    Vocabulary richness or lexical diversity measured by type-token ratio TTR,
    moving-average type-token ratio, Brunets index and Honores statistic are key
    features. TTR denotes ratio of total unique words to overall text length.
    AD disorder impacts memory resulting in AD patients potentially using more
    repetitive and less diverse vocabulary compared to healthy controls. Features
    include TTR, number of repetitive words, and number of sweepback caused by
    self-corrections.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Syntactic Complexity | Qi et al. 2023]
    Syntactic complexity of speech can be assessed through mean length of utterances,
    T-units, clauses, height of the parse tree, and statistics of Yngve depth.
    Grammatical constituents derived from parse tree analysis differentiate between
    individuals with dementia and healthy controls including frequency of different
    grammatical constituents, rate and proportion and average length of different
    phrases such as noun phrases, verb phrases and prepositional phrases. AD patients
    show difficulties in understanding the meaning of complex words and syntax,
    captured by readability features including Gunning fog index and automated
    readability index.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Acoustic vs Linguistic Performance | Qi et al. 2023]
    Linguistic features extracted from text modality consistently outperform acoustic
    features extracted from speech for AD detection. In studies on ADReSS dataset,
    acoustic-only approaches achieved approximately 74% accuracy while linguistic-only
    approaches achieved approximately 84% accuracy. Incorporating diverse features
    from multiple modalities generally leads to improved performance. The best results
    on ADReSS were achieved combining speech and text modalities reaching 93.8%
    accuracy. Model fusion approaches consistently outperformed single modality models.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Interactional and Temporal Features | Qi et al. 2023]
    During dialogue conversations temporal and interactional aspects distinguish AD
    patients from interviewers. AD patients are older people with longer lapse and
    lower speech rates. Interactional features include speech rate in syllables per
    minute, turn length in words per turn, floor control ratio indicating proportion
    of speech time by AD patients, and normalized total duration of short and long
    pauses. Using interactional features alone achieved accuracy of 87% on the
    Carolinas Conversation Collection dataset.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

    """[AD | Datasets ADReSS DementiaBank | Qi et al. 2023]
    DementiaBank is the largest publicly available database with 310 narrations from
    dementia patients and 241 from healthy controls in multiple languages. ADReSS
    derived from the Cookie session of Pitt corpus is a balanced and acoustically
    enhanced challenge dataset from Interspeech 2020 containing 78 AD patients and
    78 healthy controls matched for age and gender. ADReSSo from Interspeech 2021
    contained 166 training instances with 87 AD patients and 79 healthy controls.
    These datasets use the Cookie Theft picture description task where participants
    describe what they see in the picture, providing spontaneous speech samples.
    Source: Qi et al. (2023) Front. Aging Neurosci. 15:1224723.""",

]

# ─────────────────────────────────────────────
# COMBINED
# ─────────────────────────────────────────────

ALL_CHUNKS   = PD_CHUNKS + AD_CHUNKS
ALL_IDS      = [f"chunk_{i}" for i in range(len(ALL_CHUNKS))]
ALL_METADATA = (
    [{"disease": "PD", "index": i} for i in range(len(PD_CHUNKS))] +
    [{"disease": "AD", "index": i} for i in range(len(AD_CHUNKS))]
)

# ─────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "neuraspeech_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_collection():
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def build_knowledge_base(force_rebuild: bool = False):
    collection = get_collection()
    if collection.count() > 0 and not force_rebuild:
        print(f"Knowledge base already built ({collection.count()} chunks)")
        return
    if force_rebuild and collection.count() > 0:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        print("Cleared existing knowledge base")
    print(f"Building knowledge base with {len(ALL_CHUNKS)} chunks...")
    collection.add(
        documents=ALL_CHUNKS,
        ids=ALL_IDS,
        metadatas=ALL_METADATA
    )
    print(f"Done — PD: {len(PD_CHUNKS)}, AD: {len(AD_CHUNKS)}, Total: {len(ALL_CHUNKS)}")


def retrieve(query: str, top_k: int = 2, disease_filter: str = None) -> list:
    collection = get_collection()
    if collection.count() == 0:
        print("Warning: empty knowledge base — call build_knowledge_base() first")
        return []
    where = {"disease": disease_filter} if disease_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where=where
    )
    return results["documents"][0]


def retrieve_multi(queries: list, top_k_per_query: int = 2) -> list:
    seen   = set()
    chunks = []
    for query in queries:
        for chunk in retrieve(query, top_k=top_k_per_query):
            if chunk not in seen:
                seen.add(chunk)
                chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    build_knowledge_base(force_rebuild=True)
    print("\n--- Test Retrieval ---\n")
    tests = [
        "shimmer jitter Parkinson's vocal tremor",
        "filler words um uh Alzheimer's word retrieval",
        "silence pauses speech rate PD",
        "type token ratio vocabulary dementia",
        "acoustic vs linguistic features neurodegenerative"
    ]
    for q in tests:
        print(f"Query: {q}")
        r = retrieve(q, top_k=1)
        if r:
            print(f"  → {r[0][:120]}...")
        print()
