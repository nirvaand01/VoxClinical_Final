import { useEffect, useState } from 'react'
import { AlertCircle, Loader2 } from 'lucide-react'
import { useSamples, useSelectedSample } from '../context/SampleContext'
import { ModelResultCard } from '../components/ModelResultCard'
import { Card, CardBody } from '../components/Card'
import { StatusBadge, TypeBadge } from '../components/Badge'
import { Button } from '../components/Button'
import { createAudioUrl } from '../analysis/audioFeatures'

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m ${s}s`
}

export function AnalysisPage() {
  const { samples, selectSample, setCurrentPage, getAudioBlob } = useSamples()
  const sample = useSelectedSample()
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!sample?.hasAudio) {
      setAudioUrl(null)
      return
    }

    let url: string | null = null
    let cancelled = false

    getAudioBlob(sample.id).then((blob) => {
      if (cancelled || !blob) return
      url = createAudioUrl(blob)
      setAudioUrl(url)
    })

    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [sample?.id, sample?.hasAudio, getAudioBlob])

  if (!sample) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <p className="text-lg font-medium text-slate-700">No sample selected</p>
        <p className="mt-1 text-sm text-slate-500">Submit a new sample or pick one from history.</p>
        <div className="mt-4 flex gap-3">
          <Button onClick={() => setCurrentPage('new-sample')}>New sample</Button>
          <Button variant="secondary" onClick={() => setCurrentPage('history')}>
            View history
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-bold text-slate-900">{sample.label}</h2>
            <TypeBadge type={sample.type} />
            <StatusBadge status={sample.status} />
          </div>
          <p className="mt-1 text-sm text-slate-500">{formatDate(sample.createdAt)}</p>
        </div>
        {samples.length > 1 && (
          <select
            value={sample.id}
            onChange={(e) => selectSample(e.target.value)}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-400"
          >
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        )}
      </div>

      {sample.type === 'text' && sample.textContent && (
        <Card>
          <CardBody>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Submitted text
            </p>
            <p className="text-sm leading-relaxed text-slate-700">{sample.textContent}</p>
          </CardBody>
        </Card>
      )}

      {sample.type === 'speech' && (
        <Card>
          <CardBody className="space-y-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Audio</p>
              <p className="text-sm font-medium text-slate-800">{sample.audioFileName}</p>
              {sample.durationSeconds !== undefined && (
                <p className="text-xs text-slate-500">Duration: {formatDuration(sample.durationSeconds)}</p>
              )}
            </div>
            {audioUrl && (
              <audio controls src={audioUrl} className="w-full">
                Your browser does not support audio playback.
              </audio>
            )}
            {sample.transcript && (
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Transcript
                </p>
                <p className="text-sm leading-relaxed text-slate-700">{sample.transcript}</p>
              </div>
            )}
          </CardBody>
        </Card>
      )}

      {sample.status === 'analyzing' && (
        <div className="flex flex-col items-center py-16">
          <Loader2 className="h-10 w-10 animate-spin text-brand-600" />
          <p className="mt-4 font-medium text-slate-800">Analyzing sample...</p>
          <p className="mt-1 text-sm text-slate-500">Extracting acoustic and lexical features</p>
        </div>
      )}

      {sample.status === 'failed' && (
        <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-4">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
          <div>
            <p className="font-medium text-red-800">Analysis failed</p>
            <p className="mt-1 text-sm text-red-700">{sample.errorMessage}</p>
          </div>
        </div>
      )}

      {sample.status === 'complete' && sample.report && (
        <Card>
          <CardBody>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Clinical report
            </p>
            <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
              {sample.report}
            </div>
          </CardBody>
        </Card>
      )}

      {sample.status === 'complete' && sample.results && (
        <div className="grid gap-6 lg:grid-cols-2">
          <ModelResultCard result={sample.results.parkinson} />
          <ModelResultCard result={sample.results.dementia} />
        </div>
      )}
    </div>
  )
}
