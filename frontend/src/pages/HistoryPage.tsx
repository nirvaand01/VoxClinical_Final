import { Brain, Mic } from 'lucide-react'
import { useSamples } from '../context/SampleContext'
import { Card, CardBody, CardHeader } from '../components/Card'
import { RiskBadge, StatusBadge, TypeBadge } from '../components/Badge'
import { Button } from '../components/Button'

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function HistoryPage() {
  const { samples, selectSample, setCurrentPage, clearSamples } = useSamples()

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Sample History</h2>
          <p className="mt-1 text-slate-500">All submitted speech and text samples with analysis status.</p>
        </div>
        <div className="flex gap-2">
          {samples.length > 0 && (
            <Button variant="ghost" onClick={() => clearSamples()}>
              Clear history
            </Button>
          )}
          <Button onClick={() => setCurrentPage('new-sample')}>New sample</Button>
        </div>
      </div>

      <Card>
        <CardHeader title={`${samples.length} samples`} subtitle="Sorted by most recent first" />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-100 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <th className="px-5 py-3">Sample</th>
                <th className="px-5 py-3">Type</th>
                <th className="px-5 py-3">Date</th>
                <th className="px-5 py-3">Status</th>
                <th className="px-5 py-3">Parkinson&apos;s</th>
                <th className="px-5 py-3">Dementia</th>
                <th className="px-5 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {samples.map((sample) => (
                <tr key={sample.id} className="border-b border-slate-50 hover:bg-slate-50/80">
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2.5">
                      {sample.type === 'speech' ? (
                        <Mic className="h-4 w-4 shrink-0 text-parkinson" />
                      ) : (
                        <Brain className="h-4 w-4 shrink-0 text-als" />
                      )}
                      <span className="font-medium text-slate-900">{sample.label}</span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5">
                    <TypeBadge type={sample.type} />
                  </td>
                  <td className="px-5 py-3.5 text-slate-600">{formatDate(sample.createdAt)}</td>
                  <td className="px-5 py-3.5">
                    <StatusBadge status={sample.status} />
                  </td>
                  <td className="px-5 py-3.5">
                    {sample.results?.parkinson.riskScore != null ? (
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-parkinson">{sample.results.parkinson.riskScore}</span>
                        <RiskBadge level={sample.results.parkinson.riskLevel!} />
                      </div>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>
                  <td className="px-5 py-3.5">
                    {sample.results?.dementia.riskScore != null ? (
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-als">{sample.results.dementia.riskScore}</span>
                        <RiskBadge level={sample.results.dementia.riskLevel!} />
                      </div>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>
                  <td className="px-5 py-3.5">
                    <Button variant="ghost" onClick={() => selectSample(sample.id)}>
                      View
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {samples.length === 0 && (
          <CardBody className="py-12 text-center text-slate-500">No samples yet.</CardBody>
        )}
      </Card>
    </div>
  )
}
