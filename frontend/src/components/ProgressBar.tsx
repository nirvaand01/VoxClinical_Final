import type { RiskLevel } from '../types'

const barColors: Record<RiskLevel, string> = {
  low: 'bg-emerald-500',
  moderate: 'bg-amber-500',
  elevated: 'bg-red-500',
}

export function ProgressBar({
  value,
  max = 100,
  riskLevel,
  label,
}: {
  value: number
  max?: number
  riskLevel?: RiskLevel
  label?: string
}) {
  const pct = Math.min(100, Math.round((value / max) * 100))
  const color = riskLevel ? barColors[riskLevel] : 'bg-brand-500'

  return (
    <div>
      {label && (
        <div className="mb-1 flex justify-between text-sm">
          <span className="text-slate-600">{label}</span>
          <span className="font-medium text-slate-900">{value}{max === 100 ? '%' : ''}</span>
        </div>
      )}
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function MarkerRow({
  label,
  value,
  unit,
  description,
}: {
  label: string
  value: number
  unit: string
  description: string
}) {
  return (
    <div className="border-b border-slate-100 py-3 last:border-0">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-800">{label}</span>
        <span className="text-sm font-semibold text-slate-900">
          {value} <span className="font-normal text-slate-500">{unit}</span>
        </span>
      </div>
      <p className="mt-0.5 text-xs text-slate-500">{description}</p>
    </div>
  )
}
