import type { AddSampleInput, LinguisticMarker, ModelResult, RiskLevel, Sample } from '../types'
import { getAudioDuration } from './audioFeatures'
import { analyzeWithBackend } from './backendApi'
import { getFeatureInfo } from './featureInfo'

function riskFromScore(score: number): RiskLevel {
  if (score < 35) return 'low'
  if (score < 65) return 'moderate'
  return 'elevated'
}

function round0(n: number): number {
  return Math.round(n)
}

function round1(n: number): number {
  return Math.round(n * 10) / 10
}

function buildModelResult(opts: {
  modelName: string
  condition: ModelResult['condition']
  score: number | null
  topFeatures: Record<string, number>
  summary: string
}): ModelResult {
  const { modelName, condition, score, topFeatures, summary } = opts

  if (score === null) {
    return {
      modelName,
      condition,
      riskScore: null,
      riskLevel: null,
      markers: [],
      summary: 'Model unavailable for this sample.',
    }
  }

  const riskScore = round0(score * 100)
  const markers: LinguisticMarker[] = Object.entries(topFeatures).map(([feature, contribution]) => {
    const info = getFeatureInfo(feature)
    return {
      id: feature,
      label: info.label,
      value: round1(contribution),
      unit: 'impact',
      description: info.description,
    }
  })

  return {
    modelName,
    condition,
    riskScore,
    riskLevel: riskFromScore(riskScore),
    markers,
    summary,
  }
}

export async function analyzeSample(input: AddSampleInput): Promise<{
  durationSeconds?: number
  transcript?: string
  report?: string
  results: NonNullable<Sample['results']>
}> {
  const backend = await analyzeWithBackend(input)

  const durationSeconds = input.audioBlob ? await getAudioDuration(input.audioBlob) : undefined

  const parkinsonSummary =
    backend.prediction === 'PD'
      ? backend.reasoning_summary
      : "Primary screening signal pointed toward dementia-related patterns; Parkinson's markers were secondary."

  const dementiaSummary =
    backend.prediction === 'AD'
      ? backend.reasoning_summary
      : "Primary screening signal pointed toward Parkinson's-related patterns; dementia markers were secondary."

  const parkinson = buildModelResult({
    modelName: 'PD Risk',
    condition: "Parkinson's",
    score: backend.pd.score,
    topFeatures: backend.pd.top_features,
    summary: parkinsonSummary,
  })

  const dementia = buildModelResult({
    modelName: 'AD Risk',
    condition: 'Dementia',
    score: backend.ad.score,
    topFeatures: backend.ad.top_features,
    summary: dementiaSummary,
  })

  return {
    durationSeconds,
    transcript: backend.transcript || input.transcript,
    report: backend.report,
    results: { parkinson, dementia },
  }
}

export async function getDurationFromBlob(blob: Blob): Promise<number> {
  return getAudioDuration(blob)
}
