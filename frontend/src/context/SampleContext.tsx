import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import type { AddSampleInput, PageId, Sample } from '../types'
import { analyzeSample } from '../analysis/analyzeSample'
import { clearAllSamples, loadAudioBlob, loadSamples, saveAudioBlob, saveSamples } from '../analysis/storage'

interface SampleContextValue {
  samples: Sample[]
  selectedSampleId: string | null
  currentPage: PageId
  setCurrentPage: (page: PageId) => void
  selectSample: (id: string | null) => void
  addSample: (input: AddSampleInput) => Promise<string>
  getAudioBlob: (sampleId: string) => Promise<Blob | null>
  clearSamples: () => Promise<void>
}

const SampleContext = createContext<SampleContextValue | null>(null)

const audioCache = new Map<string, Blob>()

export function SampleProvider({ children }: { children: ReactNode }) {
  const [samples, setSamples] = useState<Sample[]>(() => loadSamples())
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState<PageId>('dashboard')

  useEffect(() => {
    saveSamples(samples)
  }, [samples])

  const selectSample = (id: string | null) => {
    setSelectedSampleId(id)
    if (id) setCurrentPage('analysis')
  }

  const getAudioBlob = useCallback(async (sampleId: string): Promise<Blob | null> => {
    if (audioCache.has(sampleId)) return audioCache.get(sampleId)!
    const blob = await loadAudioBlob(sampleId)
    if (blob) audioCache.set(sampleId, blob)
    return blob
  }, [])

  const addSample: SampleContextValue['addSample'] = async (input) => {
    const id = `sample-${Date.now()}`
    const pending: Sample = {
      id,
      type: input.type,
      label: input.label,
      textContent: input.textContent,
      transcript: input.transcript,
      audioFileName: input.audioFileName,
      hasAudio: !!input.audioBlob,
      createdAt: new Date().toISOString(),
      status: 'analyzing',
    }

    if (input.audioBlob) {
      audioCache.set(id, input.audioBlob)
      await saveAudioBlob(id, input.audioBlob)
    }

    setSamples((prev) => [pending, ...prev])
    setSelectedSampleId(id)
    setCurrentPage('analysis')

    try {
      const analysis = await analyzeSample(input)
      setSamples((prev) =>
        prev.map((s) =>
          s.id === id
            ? {
                ...s,
                status: 'complete',
                durationSeconds: analysis.durationSeconds,
                transcript: analysis.transcript ?? s.transcript,
                report: analysis.report,
                results: analysis.results,
              }
            : s,
        ),
      )
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed.'
      setSamples((prev) =>
        prev.map((s) =>
          s.id === id ? { ...s, status: 'failed', errorMessage: message } : s,
        ),
      )
    }

    return id
  }

  const clearSamples = async () => {
    await clearAllSamples()
    audioCache.clear()
    setSelectedSampleId(null)
    setSamples([])
  }

  return (
    <SampleContext.Provider
      value={{
        samples,
        selectedSampleId,
        currentPage,
        setCurrentPage,
        selectSample,
        addSample,
        getAudioBlob,
        clearSamples,
      }}
    >
      {children}
    </SampleContext.Provider>
  )
}

export function useSamples() {
  const ctx = useContext(SampleContext)
  if (!ctx) throw new Error('useSamples must be used within SampleProvider')
  return ctx
}

export function useSelectedSample(): Sample | undefined {
  const { samples, selectedSampleId } = useSamples()
  return samples.find((s) => s.id === selectedSampleId)
}
