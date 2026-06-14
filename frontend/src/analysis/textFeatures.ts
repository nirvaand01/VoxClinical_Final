import type { SampleType } from '../types'

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
  'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
  'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
  'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
])

const PRONOUNS = new Set([
  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
  'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
  'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
])

export interface TextFeatures {
  wordCount: number
  sentenceCount: number
  uniqueWordCount: number
  typeTokenRatio: number
  pronounRatio: number
  avgWordsPerSentence: number
  contentWordRatio: number
  coherenceScore: number
  ideaDensity: number
}

function tokenizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9'\s-]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 0)
}

function splitSentences(text: string): string[] {
  return text
    .split(/[.!?]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

function sentenceKeywords(sentence: string): Set<string> {
  return new Set(
    tokenizeWords(sentence).filter((w) => w.length > 2 && !STOP_WORDS.has(w)),
  )
}

function computeCoherence(sentences: string[]): number {
  if (sentences.length < 2) return 100

  let overlapSum = 0
  for (let i = 1; i < sentences.length; i++) {
    const prev = sentenceKeywords(sentences[i - 1])
    const curr = sentenceKeywords(sentences[i])
    if (prev.size === 0 || curr.size === 0) continue
    let shared = 0
    for (const word of prev) {
      if (curr.has(word)) shared++
    }
    overlapSum += shared / Math.max(prev.size, curr.size)
  }

  return Math.round((overlapSum / (sentences.length - 1)) * 100)
}

export function extractTextFeatures(text: string): TextFeatures {
  const words = tokenizeWords(text)
  const sentences = splitSentences(text)
  const uniqueWords = new Set(words)
  const pronounCount = words.filter((w) => PRONOUNS.has(w)).length
  const contentWords = words.filter((w) => !STOP_WORDS.has(w))

  const wordCount = words.length
  const sentenceCount = Math.max(sentences.length, 1)

  return {
    wordCount,
    sentenceCount,
    uniqueWordCount: uniqueWords.size,
    typeTokenRatio: wordCount > 0 ? uniqueWords.size / wordCount : 0,
    pronounRatio: wordCount > 0 ? (pronounCount / wordCount) * 100 : 0,
    avgWordsPerSentence: wordCount / sentenceCount,
    contentWordRatio: wordCount > 0 ? (contentWords.length / wordCount) * 100 : 0,
    coherenceScore: computeCoherence(sentences),
    ideaDensity: contentWords.length / sentenceCount,
  }
}

export function getAnalysisText(sample: {
  type: SampleType
  textContent?: string
  transcript?: string
}): string | null {
  if (sample.type === 'text') {
    return sample.textContent?.trim() || null
  }
  return sample.transcript?.trim() || null
}
