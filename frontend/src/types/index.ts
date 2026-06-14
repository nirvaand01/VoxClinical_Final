export type SampleType = 'speech' | 'text'

export type RiskLevel = 'low' | 'moderate' | 'elevated'

export type SampleStatus = 'pending' | 'analyzing' | 'complete' | 'failed'

export interface LinguisticMarker {
  id: string
  label: string
  value: number
  unit: string
  description: string
}

export interface ModelResult {
  modelName: string
  condition: "Parkinson's" | 'Dementia'
  riskScore: number | null
  riskLevel: RiskLevel | null
  markers: LinguisticMarker[]
  summary: string
}

export interface Sample {
  id: string
  type: SampleType
  label: string
  textContent?: string
  transcript?: string
  audioFileName?: string
  hasAudio?: boolean
  durationSeconds?: number
  createdAt: string
  status: SampleStatus
  errorMessage?: string
  results?: {
    parkinson: ModelResult
    dementia: ModelResult
  }
  report?: string
}

export type PageId = 'dashboard' | 'new-sample' | 'analysis' | 'history'

export interface AddSampleInput {
  type: SampleType
  label: string
  textContent?: string
  transcript?: string
  audioBlob?: Blob
  audioFileName?: string
}
