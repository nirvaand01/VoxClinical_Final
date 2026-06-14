import { ArrowRight, Brain, Mic, PlusCircle } from 'lucide-react'
import { useSamples } from '../context/SampleContext'
import { Card, CardBody, CardHeader } from '../components/Card'
import { Button } from '../components/Button'
import { RiskBadge, TypeBadge } from '../components/Badge'
import { ModelResultCard } from '../components/ModelResultCard'

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function DashboardPage() {
  const { samples, setCurrentPage, selectSample } = useSamples()
  const latest = samples.find((s) => s.status === 'complete')
  const recent = samples.slice(0, 4)

  if (samples.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-brand-50">
          <Mic className="h-8 w-8 text-brand-600" />
        </div>
        <h2 className="mt-4 text-xl font-bold text-slate-900">No samples yet</h2>
        <p className="mt-2 max-w-md text-sm text-slate-500">
          Submit a speech recording or text sample to analyze linguistic markers for
          Parkinson&apos;s and Dementia.
        </p>
        <Button className="mt-6" onClick={() => setCurrentPage('new-sample')}>
          <PlusCircle className="h-4 w-4" />
          Submit your first sample
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Dashboard</h2>
          <p className="mt-1 text-slate-500">
            Track linguistic markers across speech and text samples.
          </p>
        </div>
        <Button onClick={() => setCurrentPage('new-sample')}>
          <Mic className="h-4 w-4" />
          New sample
        </Button>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardBody>
            <p className="text-sm text-slate-500">Total samples</p>
            <p className="mt-1 text-3xl font-bold text-slate-900">{samples.length}</p>
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <p className="text-sm text-slate-500">Speech samples</p>
            <p className="mt-1 text-3xl font-bold text-parkinson">
              {samples.filter((s) => s.type === 'speech').length}
            </p>
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <p className="text-sm text-slate-500">Text samples</p>
            <p className="mt-1 text-3xl font-bold text-als">
              {samples.filter((s) => s.type === 'text').length}
            </p>
          </CardBody>
        </Card>
      </div>

      {latest?.results && (
        <div>
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-900">Latest analysis</h3>
            <Button variant="ghost" onClick={() => selectSample(latest.id)}>
              View details <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <ModelResultCard result={latest.results.parkinson} />
            <ModelResultCard result={latest.results.dementia} />
          </div>
        </div>
      )}

      <Card>
        <CardHeader
          title="Recent samples"
          subtitle="Click a sample to view full analysis"
          action={
            <Button variant="ghost" onClick={() => setCurrentPage('history')}>
              View all
            </Button>
          }
        />
        {recent.length > 0 ? (
          <CardBody className="divide-y divide-slate-100 p-0">
            {recent.map((sample) => (
              <button
                key={sample.id}
                onClick={() => selectSample(sample.id)}
                className="flex w-full items-center justify-between px-5 py-3.5 text-left transition-colors hover:bg-slate-50"
              >
                <div className="flex items-center gap-3">
                  {sample.type === 'speech' ? (
                    <Mic className="h-4 w-4 text-parkinson" />
                  ) : (
                    <Brain className="h-4 w-4 text-als" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-slate-900">{sample.label}</p>
                    <p className="text-xs text-slate-500">{formatDate(sample.createdAt)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <TypeBadge type={sample.type} />
                  {sample.results?.parkinson.riskLevel && (
                    <RiskBadge level={sample.results.parkinson.riskLevel} />
                  )}
                  {sample.results?.dementia.riskLevel && (
                    <RiskBadge level={sample.results.dementia.riskLevel} />
                  )}
                </div>
              </button>
            ))}
          </CardBody>
        ) : (
          <CardBody className="py-8 text-center text-sm text-slate-500">
            No samples to show.
          </CardBody>
        )}
      </Card>
    </div>
  )
}
