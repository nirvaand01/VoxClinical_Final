export interface AudioFeatures {
  durationSeconds: number
  pauseCount: number
  pauseFrequencyPerMinute: number
  avgPauseDurationMs: number
  prosodyVariation: number
}

const PAUSE_THRESHOLD = 0.02
const MIN_PAUSE_MS = 250

export async function getAudioDuration(blob: Blob): Promise<number> {
  const arrayBuffer = await blob.arrayBuffer()
  const audioContext = new AudioContext()
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0))
    return audioBuffer.duration
  } finally {
    await audioContext.close()
  }
}

export async function extractAudioFeatures(blob: Blob): Promise<AudioFeatures> {
  const arrayBuffer = await blob.arrayBuffer()
  const audioContext = new AudioContext()
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0))
    const channelData = audioBuffer.getChannelData(0)
    const sampleRate = audioBuffer.sampleRate
    const durationSeconds = audioBuffer.duration

    const windowSize = Math.floor(sampleRate * 0.05)
    const rmsValues: number[] = []

    for (let i = 0; i < channelData.length; i += windowSize) {
      let sum = 0
      const end = Math.min(i + windowSize, channelData.length)
      for (let j = i; j < end; j++) sum += channelData[j] * channelData[j]
      rmsValues.push(Math.sqrt(sum / (end - i)))
    }

    const pauseMinWindows = Math.ceil(MIN_PAUSE_MS / 50)
    let pauseCount = 0
    let pauseWindows = 0
    let pauseDurationMs = 0

    for (let i = 0; i < rmsValues.length; i++) {
      if (rmsValues[i] < PAUSE_THRESHOLD) {
        pauseWindows++
      } else if (pauseWindows >= pauseMinWindows) {
        pauseCount++
        pauseDurationMs += pauseWindows * 50
        pauseWindows = 0
      } else {
        pauseWindows = 0
      }
    }

    const mean = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length
    const variance =
      rmsValues.reduce((sum, v) => sum + (v - mean) ** 2, 0) / rmsValues.length
    const prosodyVariation = Math.min(100, Math.round(Math.sqrt(variance) * 500))

    const durationMinutes = durationSeconds / 60

    return {
      durationSeconds: Math.round(durationSeconds * 10) / 10,
      pauseCount,
      pauseFrequencyPerMinute:
        durationMinutes > 0 ? Math.round((pauseCount / durationMinutes) * 10) / 10 : 0,
      avgPauseDurationMs:
        pauseCount > 0 ? Math.round(pauseDurationMs / pauseCount) : 0,
      prosodyVariation,
    }
  } finally {
    await audioContext.close()
  }
}

export function createAudioUrl(blob: Blob): string {
  return URL.createObjectURL(blob)
}
