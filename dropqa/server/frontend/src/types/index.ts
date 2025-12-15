/** 来源信息 */
export interface Source {
  document_name: string
  path: string
  content_snippet: string
  score?: number
}

/** 推理步骤 */
export interface ReasoningStep {
  step: string
  action: string
  result?: string
}

/** QA 响应 */
export interface QAResponse {
  answer: string
  sources: Source[]
  mode: 'simple' | 'agentic'
  reasoning_trace?: ReasoningStep[]
  rewritten_query?: {
    original: string
    rewritten: string
    reasoning: string
  }
}

/** 历史记录项 */
export interface HistoryItem {
  question: string
  answer: string
  sources: Source[]
  timestamp: Date
}
