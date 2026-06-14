import type { ModelResult } from '../types'
import { Card, CardBody, CardHeader } from './Card'
import { RiskBadge } from './Badge'
import { MarkerRow, ProgressBar } from './ProgressBar'

const accentColors = {
  "Parkinson's": 'border-l-parkinson',
  Dementia: 'border-l-als',
}

const scoreColors = {
  "Parkinson's": 'text-parkinson',
  Dementia: 'text-als',
}

export function ModelResultCard({ result }: { result: ModelResult }) {
  const hasScore = result.riskScore !== null

  return (
    <Card className={`border-l-4 ${accentColors[result.condition]}`}>
      <CardHeader
        title={result.modelName}
        subtitle={`${result.condition} markers`}
        action={result.riskLevel ? <RiskBadge level={result.riskLevel} /> : undefined}
      />
      <CardBody>
        {hasScore ? (
          <>
            <div className="mb-5">
              <p className="text-sm text-slate-500">Risk score</p>
              <p className={`text-4xl font-bold ${scoreColors[result.condition]}`}>
                {result.riskScore}
              </p>
            </div>

            <ProgressBar
              value={result.riskScore!}
              riskLevel={result.riskLevel!}
              label="Composite risk index"
            />
          </>
        ) : (
          <p className="mb-4 text-sm text-slate-500">Not enough data to compute a risk score.</p>
        )}

        <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2.5 text-sm leading-relaxed text-slate-600">
          {result.summary}
        </p>

        {result.markers.length > 0 && (
          <div className="mt-5">
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Extracted markers
            </h4>
            {result.markers.map((m) => (
              <MarkerRow key={m.id} {...m} />
            ))}
          </div>
        )}
      </CardBody>
    </Card>
  )
}
