import type { AddSampleInput } from '../types'
import { getAnalysisText } from './textFeatures'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export interface BackendDiseaseResult {
  score: number | null
  top_features: Record<string, number>
}

export interface BackendAnalysis {
  report: string
  prediction: 'PD' | 'AD'
  confidence: number
  dominant_signal: 'acoustic' | 'linguistic'
  key_evidence: string[]
  reasoning_summary: string
  transcript: string | null
  pd: BackendDiseaseResult
  ad: BackendDiseaseResult
}

export async function analyzeWithBackend(input: AddSampleInput): Promise<BackendAnalysis> {
  const form = new FormData()

  if (input.audioBlob) {
    form.append('audio', input.audioBlob, input.audioFileName ?? 'audio.wav')
  }

  const text = getAnalysisText(input)
  if (text) form.append('text', text)

  let res: Response
  try {
    res = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: form })
  } catch {
    throw new Error(`Could not reach the analysis server at ${API_URL}. Is it running?`)
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Analysis request failed (${res.status})`)
  }

  return res.json()
}
