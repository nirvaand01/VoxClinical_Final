import { SampleProvider, useSamples } from './context/SampleContext'
import { Layout } from './components/Layout'
import { DashboardPage } from './pages/DashboardPage'
import { NewSamplePage } from './pages/NewSamplePage'
import { AnalysisPage } from './pages/AnalysisPage'
import { HistoryPage } from './pages/HistoryPage'

function PageRouter() {
  const { currentPage } = useSamples()

  switch (currentPage) {
    case 'dashboard':
      return <DashboardPage />
    case 'new-sample':
      return <NewSamplePage />
    case 'analysis':
      return <AnalysisPage />
    case 'history':
      return <HistoryPage />
  }
}

function App() {
  return (
    <SampleProvider>
      <Layout>
        <PageRouter />
      </Layout>
    </SampleProvider>
  )
}

export default App
