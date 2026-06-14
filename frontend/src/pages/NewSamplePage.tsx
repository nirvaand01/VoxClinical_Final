import { useRef, useState } from 'react'
import { AlertCircle, FileAudio, Loader2, Mic, Square, Type, Upload } from 'lucide-react'
import { useSamples } from '../context/SampleContext'
import { Card, CardBody, CardHeader } from '../components/Card'
import { Button } from '../components/Button'

type InputMode = 'speech-upload' | 'speech-record' | 'text'

const MAX_FILE_SIZE = 10 * 1024 * 1024

export function NewSamplePage() {
  const { addSample } = useSamples()
  const [mode, setMode] = useState<InputMode>('speech-upload')
  const [label, setLabel] = useState('')
  const [textContent, setTextContent] = useState('')
  const [transcript, setTranscript] = useState('')
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const transcriptPartsRef = useRef<string[]>([])

  const resetSpeechState = () => {
    setAudioFile(null)
    setTranscript('')
    transcriptPartsRef.current = []
  }

  const startRecording = async () => {
    setError(null)
    resetSpeechState()

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioFile(new File([blob], `recording_${Date.now()}.webm`, { type: 'audio/webm' }))
        setTranscript(transcriptPartsRef.current.join(' ').trim())
        stream.getTracks().forEach((t) => t.stop())
      }

      recorder.start()
      mediaRecorderRef.current = recorder

      const SpeechRecognition =
        window.SpeechRecognition ?? window.webkitSpeechRecognition

      if (SpeechRecognition) {
        const recognition = new SpeechRecognition()
        recognition.continuous = true
        recognition.interimResults = true
        recognition.lang = 'en-US'
        recognition.onresult = (event) => {
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
              transcriptPartsRef.current.push(event.results[i][0].transcript.trim())
            }
          }
        }
        recognition.start()
        recognitionRef.current = recognition
      }

      setIsRecording(true)
    } catch {
      setError('Microphone access denied. Allow microphone access or upload a file instead.')
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    recognitionRef.current?.stop()
    setIsRecording(false)
  }

  const handleFileChange = (file: File | null) => {
    setError(null)
    if (!file) {
      setAudioFile(null)
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('File exceeds 10 MB limit.')
      setAudioFile(null)
      return
    }
    setAudioFile(file)
  }

  const handleSubmit = async () => {
    if (!label.trim()) return
    setError(null)
    setIsSubmitting(true)

    try {
      if (mode === 'text') {
        await addSample({ type: 'text', label: label.trim(), textContent: textContent.trim() })
      } else {
        if (!audioFile) {
          setError('An audio file is required.')
          return
        }
        await addSample({
          type: 'speech',
          label: label.trim(),
          audioBlob: audioFile,
          audioFileName: audioFile.name,
          transcript: transcript.trim() || undefined,
        })
      }

      setLabel('')
      setTextContent('')
      resetSpeechState()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit sample.')
    } finally {
      setIsSubmitting(false)
    }
  }

const canSubmit =
    label.trim() &&
    !isSubmitting &&
    (mode === 'text' ? textContent.trim().length >= 20 : !!audioFile)

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">New Sample</h2>
        <p className="mt-1 text-slate-500">
          Submit a speech recording or text sample for analysis.
        </p>
      </div>

      <div className="flex gap-2">
        {(
          [
            { id: 'speech-upload' as const, label: 'Upload audio', icon: Upload },
            { id: 'speech-record' as const, label: 'Record', icon: Mic },
            { id: 'text' as const, label: 'Text input', icon: Type },
          ] as const
        ).map(({ id, label: tabLabel, icon: Icon }) => (
          <button
            key={id}
            onClick={() => {
              setMode(id)
              setError(null)
              if (id === 'text') resetSpeechState()
            }}
            className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-3 py-2.5 text-sm font-medium transition-colors ${
              mode === id
                ? 'border-brand-300 bg-brand-50 text-brand-700'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
            }`}
          >
            <Icon className="h-4 w-4" />
            {tabLabel}
          </button>
        ))}
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      <Card>
        <CardHeader title="Sample details" subtitle="Provide a label and your sample content" />
        <CardBody className="space-y-4">
          <div>
            <label htmlFor="sample-label" className="mb-1.5 block text-sm font-medium text-slate-700">
              Sample label
            </label>
            <input
              id="sample-label"
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g. Weekly check-in — March 12"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
            />
          </div>

          {mode === 'speech-upload' && (
            <>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-slate-700">Audio file</label>
                <label className="flex cursor-pointer flex-col items-center rounded-lg border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-8 transition-colors hover:border-brand-300 hover:bg-brand-50/30">
                  <FileAudio className="mb-2 h-8 w-8 text-slate-400" />
                  <span className="text-sm font-medium text-slate-700">
                    {audioFile ? audioFile.name : 'Click to upload .wav, .mp3, or .webm'}
                  </span>
                  <span className="mt-1 text-xs text-slate-500">Max 10 MB</span>
                  <input
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
                  />
                </label>
              </div>
              <div>
                <label htmlFor="transcript" className="mb-1.5 block text-sm font-medium text-slate-700">
                  Transcript <span className="font-normal text-slate-400">(recommended for full analysis)</span>
                </label>
                <textarea
                  id="transcript"
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  rows={4}
                  placeholder="Paste or type what was said in the recording..."
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
                />
              </div>
            </>
          )}

          {mode === 'speech-record' && (
            <div className="space-y-4">
              <div className="flex flex-col items-center rounded-lg border border-slate-200 bg-slate-50 px-6 py-8">
                <div
                  className={`mb-4 flex h-16 w-16 items-center justify-center rounded-full ${isRecording ? 'animate-pulse bg-red-100' : 'bg-brand-100'}`}
                >
                  <Mic className={`h-8 w-8 ${isRecording ? 'text-red-600' : 'text-brand-600'}`} />
                </div>
                {isRecording ? (
                  <Button variant="danger" onClick={stopRecording}>
                    <Square className="h-4 w-4" />
                    Stop recording
                  </Button>
                ) : (
                  <Button onClick={startRecording}>
                    <Mic className="h-4 w-4" />
                    Start recording
                  </Button>
                )}
                {audioFile && (
                  <p className="mt-3 text-sm text-emerald-600">Recording saved: {audioFile.name}</p>
                )}
                <p className="mt-2 text-xs text-slate-500">
                  Speech is transcribed live when your browser supports it. You can edit the transcript below.
                </p>
              </div>
              {(audioFile || transcript) && (
                <div>
                  <label htmlFor="record-transcript" className="mb-1.5 block text-sm font-medium text-slate-700">
                    Transcript
                  </label>
                  <textarea
                    id="record-transcript"
                    value={transcript}
                    onChange={(e) => setTranscript(e.target.value)}
                    rows={4}
                    placeholder="Transcript will appear here after recording..."
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
                  />
                </div>
              )}
            </div>
          )}

          {mode === 'text' && (
            <div>
              <label htmlFor="text-content" className="mb-1.5 block text-sm font-medium text-slate-700">
                Text content
              </label>
              <textarea
                id="text-content"
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                rows={6}
                placeholder="Enter spontaneous speech transcript, picture description, or narrative text..."
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100"
              />
              <p className="mt-1 text-xs text-slate-500">
                {textContent.length} characters · minimum 20 required
              </p>
            </div>
          )}

          <Button className="w-full" disabled={!canSubmit} onClick={handleSubmit}>
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze sample'
            )}
          </Button>
        </CardBody>
      </Card>
    </div>
  )
}
