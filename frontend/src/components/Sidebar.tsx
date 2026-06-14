import {
  Brain,
  Clock,
  LayoutDashboard,
  Mic,
  PlusCircle,
} from 'lucide-react'
import type { PageId } from '../types'
import { useSamples } from '../context/SampleContext'

const navItems: { id: PageId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'new-sample', label: 'New Sample', icon: PlusCircle },
  { id: 'analysis', label: 'Analysis', icon: Brain },
  { id: 'history', label: 'History', icon: Clock },
]

export function Sidebar() {
  const { currentPage, setCurrentPage, samples } = useSamples()
  const completeCount = samples.filter((s) => s.status === 'complete').length

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-5 py-5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white">
            <Mic className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-900">VoxClinical</h1>
            <p className="text-xs text-slate-500">Voice &amp; linguistic analysis</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setCurrentPage(id)}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
              currentPage === id
                ? 'bg-brand-50 text-brand-700'
                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </nav>

      <div className="border-t border-slate-100 px-5 py-4">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Session stats</p>
        <p className="mt-1 text-2xl font-bold text-slate-900">{completeCount}</p>
        <p className="text-xs text-slate-500">completed analyses</p>
      </div>
    </aside>
  )
}

export function DisclaimerBanner() {
  return (
    <div className="border-b border-amber-200 bg-amber-50 px-6 py-2.5 text-center text-xs text-amber-800">
      <strong>Research prototype.</strong> Not a medical device — for screening research only. Do not use for clinical diagnosis.
    </div>
  )
}
