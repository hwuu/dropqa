import { ref } from 'vue'
import type { Source, ReasoningStep, QAResponse } from '@/types'

export function useQA() {
  const answer = ref('')
  const sources = ref<Source[]>([])
  const loading = ref(false)
  const error = ref('')
  const mode = ref<'simple' | 'agentic'>('simple')
  const reasoningTrace = ref<ReasoningStep[]>([])
  const progress = ref('')

  async function ask(question: string) {
    if (!question.trim()) return

    loading.value = true
    error.value = ''
    answer.value = ''
    sources.value = []
    reasoningTrace.value = []
    progress.value = '正在连接...'

    try {
      const response = await fetch('/api/qa/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `请求失败 (${response.status})`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('无法读取响应流')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // 解析 SSE 事件
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let eventType = ''
        let eventData = ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7)
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6)
          } else if (line === '' && eventType && eventData) {
            // 完整事件，处理它
            try {
              const data = JSON.parse(eventData)

              if (eventType === 'progress') {
                progress.value = data.message
              } else if (eventType === 'complete') {
                answer.value = data.answer
                sources.value = data.sources || []
                mode.value = data.mode || 'simple'
                reasoningTrace.value = data.reasoning_trace || []
                progress.value = ''
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }

            eventType = ''
            eventData = ''
          }
        }
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : '未知错误'
      progress.value = ''
    } finally {
      loading.value = false
    }
  }

  function reset() {
    answer.value = ''
    sources.value = []
    loading.value = false
    error.value = ''
    mode.value = 'simple'
    reasoningTrace.value = []
    progress.value = ''
  }

  return {
    answer,
    sources,
    loading,
    error,
    mode,
    reasoningTrace,
    progress,
    ask,
    reset,
  }
}
