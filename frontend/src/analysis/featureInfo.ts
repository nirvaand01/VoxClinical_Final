export interface FeatureInfo {
  label: string
  description: string
}

const MFCC_FAMILY: Record<string, { name: string; description: string }> = {
  mfcc: {
    name: 'Vocal Timbre',
    description:
      'Reflects the overall tone and resonance of the voice. Atypical patterns can indicate changes in how the throat, mouth, and tongue shape sound during speech.',
  },
  dmfcc: {
    name: 'Vocal Timbre Shift Rate',
    description:
      'Reflects how quickly the tone and resonance of the voice change between sounds, related to the speed of articulatory movements.',
  },
  d2mfcc: {
    name: 'Vocal Timbre Shift Acceleration',
    description:
      'Reflects acceleration in how the voice changes between sounds, sensitive to fine motor control of the tongue, lips, and jaw.',
  },
}

const EXPLICIT_FEATURES: Record<string, FeatureInfo> = {
  ac_f0_mean: {
    label: 'Average Pitch (F0)',
    description:
      'The average pitch of the voice. A reduced pitch range and monotone speech are characteristic of hypokinetic (Parkinsonian) speech patterns.',
  },
  ac_f0_jitter: {
    label: 'Pitch Jitter',
    description:
      'Cycle-to-cycle variation in vocal pitch. Increased jitter can indicate reduced control over vocal fold vibration.',
  },
  ac_shimmer_approx: {
    label: 'Amplitude Shimmer',
    description:
      'Cycle-to-cycle variation in vocal loudness. Increased shimmer reflects vocal instability, often linked to reduced respiratory or laryngeal control.',
  },
  ac_spec_flatness_mean: {
    label: 'Spectral Flatness',
    description:
      'Measures how noise-like versus tonal the voice is. A breathier, less resonant voice produces a flatter spectrum.',
  },
  ac_mel_mean: {
    label: 'Voice Energy Distribution',
    description:
      'Overall energy distribution across perceptually-scaled frequency bands. Lower energy can reflect reduced vocal loudness (hypophonia).',
  },
  ac_pause_ratio: {
    label: 'Pause Ratio',
    description:
      'The proportion of the recording that is silence. Increased pausing can reflect word-finding difficulty or slowed speech initiation.',
  },
  li_sent_len_cv: {
    label: 'Sentence Length Variability',
    description:
      'How much sentence length varies across the sample. Reduced variability can reflect simplified or repetitive sentence structure.',
  },
  li_std_gap: {
    label: 'Pause Duration Variability',
    description:
      'How consistent the pauses between words are. Irregular pausing can reflect word-finding difficulty or disrupted speech planning.',
  },
  li_rhythm_regularity: {
    label: 'Speech Rhythm Regularity',
    description:
      'How evenly-timed speech is. Both an overly rigid (monotone) and an overly irregular rhythm can be markers of speech-motor changes.',
  },
  li_max_word_len: {
    label: 'Longest Word Used',
    description:
      'The length of the longest word produced. A shorter maximum word length can reflect simplified vocabulary or word-finding difficulty.',
  },
  li_bigram_rep: {
    label: 'Word-Pair Repetition',
    description:
      'How often two-word sequences are repeated. Increased repetition can reflect perseveration, a pattern associated with cognitive decline.',
  },
  li_long_pause_count: {
    label: 'Long Pause Count',
    description:
      'The number of unusually long pauses during speech. Frequent long pauses can reflect word-finding difficulty or slowed cognitive processing.',
  },
  li_pronoun_ratio: {
    label: 'Pronoun Usage Ratio',
    description:
      'The proportion of words that are pronouns. Increased use of vague pronouns ("it", "this", "that") in place of specific nouns can reflect word-finding difficulty.',
  },
  syn_verb_ratio: {
    label: 'Verb Usage Ratio',
    description:
      'The proportion of words that are verbs. Reduced verb usage has been linked to grammatical simplification in some forms of cognitive decline.',
  },
  syn_mean_parse_depth: {
    label: 'Sentence Structure Complexity',
    description:
      'How deeply nested the sentence structures are. Shallower structures can reflect reduced syntactic complexity, often seen with cognitive decline.',
  },
  freq_mean_log_freq: {
    label: 'Word Familiarity',
    description:
      'How common the words used are in everyday language. A preference for more common words can reflect word-finding difficulty.',
  },
  sem_avg_synset_depth: {
    label: 'Semantic Specificity',
    description:
      'How specific versus generic the nouns used are. Reliance on more generic terms can reflect semantic memory decline.',
  },
  read_avg_syllables_per_word: {
    label: 'Word Length (Syllables)',
    description:
      'The average number of syllables per word. Shorter, simpler words can reflect reduced lexical access.',
  },
  read_flesch_reading_ease: {
    label: 'Reading Ease Score',
    description:
      'A standardized measure of how simple the sentence and word structure is. Higher scores indicate simpler language, which can reflect reduced linguistic complexity.',
  },
}

function titleCase(s: string): string {
  return s.replace(/\b\w/g, (c) => c.toUpperCase())
}

export function getFeatureInfo(name: string): FeatureInfo {
  if (name in EXPLICIT_FEATURES) return EXPLICIT_FEATURES[name]

  const mfccMatch = name.match(/^ac_(mfcc|dmfcc|d2mfcc)(\d+)_(mean|std)$/)
  if (mfccMatch) {
    const [, family, coef, stat] = mfccMatch
    const info = MFCC_FAMILY[family]
    const variant = stat === 'mean' ? 'average' : 'variability'
    return {
      label: `${info.name} #${coef} (${variant})`,
      description: info.description,
    }
  }

  const contrastMatch = name.match(/^ac_spec_contrast_b(\d+)$/)
  if (contrastMatch) {
    return {
      label: `Spectral Contrast (band ${contrastMatch[1]})`,
      description:
        'The balance between peaks and valleys in the voice spectrum at this frequency range, related to voice clarity and resonance.',
    }
  }

  const centroidMatch = name.match(/^ac_spec_centroid_(mean|std)$/)
  if (centroidMatch) {
    const variant = centroidMatch[1] === 'mean' ? 'average' : 'variability'
    return {
      label: `Spectral Brightness (${variant})`,
      description:
        'Reflects the perceived brightness of the voice. Lower values can indicate a duller, breathier voice quality, sometimes seen in hypokinetic speech.',
    }
  }

  // Fallback: humanize the raw feature name
  const label = titleCase(
    name
      .replace(/^ac_/, '')
      .replace(/^(li_|syn_|freq_|sem_|read_)/, '')
      .replace(/_/g, ' '),
  )
  return {
    label,
    description: 'A speech marker analyzed as part of this assessment.',
  }
}
