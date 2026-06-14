import type { RiskLevel } from '../types'

const styles: Record<RiskLevel, string> = {
  low: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  moderate: 'bg-amber-50 text-amber-700 ring-amber-200',
  elevated: 'bg-red-50 text-red-700 ring-red-200',
}

const labels: Record<RiskLevel, string> = {
  low: 'Low',
  moderate: 'Moderate',
  elevated: 'Elevated',
}

export function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${styles[level]}`}
    >
      {labels[level]} risk
    </span>
  )
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    complete: 'bg-emerald-50 text-emerald-700',
    analyzing: 'bg-blue-50 text-blue-700',
    pending: 'bg-slate-100 text-slate-600',
    failed: 'bg-red-50 text-red-700',
  }
  return (
    <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${map[status] ?? map.pending}`}>
      {status}
    </span>
  )
}

export function TypeBadge({ type }: { type: 'speech' | 'text' }) {
  return (
    <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium capitalize text-slate-600">
      {type}
    </span>
  )
}
