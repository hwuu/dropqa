import { ref, watch } from 'vue'
import type { HistoryItem } from '@/types'

const STORAGE_KEY = 'dropqa-history'
const MAX_HISTORY = 50

export function useHistory() {
  const history = ref<HistoryItem[]>([])

  // Load from localStorage
  function loadHistory() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored)
        history.value = parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp),
        }))
      }
    } catch {
      history.value = []
    }
  }

  // Save to localStorage
  function saveHistory() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history.value))
    } catch {
      // Ignore storage errors
    }
  }

  // Add item to history
  function addToHistory(item: HistoryItem) {
    // Avoid duplicates (same question within last 5 items)
    const recentQuestions = history.value.slice(0, 5).map(h => h.question)
    if (recentQuestions.includes(item.question)) {
      return
    }

    history.value.unshift(item)

    // Limit history size
    if (history.value.length > MAX_HISTORY) {
      history.value = history.value.slice(0, MAX_HISTORY)
    }

    saveHistory()
  }

  // Clear all history
  function clearHistory() {
    history.value = []
    saveHistory()
  }

  // Initialize
  loadHistory()

  return {
    history,
    addToHistory,
    clearHistory,
  }
}
