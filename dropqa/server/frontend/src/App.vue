<script setup lang="ts">
import { ref } from 'vue'
import SearchBox from './components/SearchBox.vue'
import AnswerCard from './components/AnswerCard.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import { useQA } from './composables/useQA'
import { useHistory } from './composables/useHistory'

const { answer, sources, loading, error, mode, reasoningTrace, progress, ask } = useQA()
const { history, addToHistory, clearHistory } = useHistory()
const showHistory = ref(false)

async function handleAsk(question: string) {
  await ask(question)
  if (answer.value && !error.value) {
    addToHistory({
      question,
      answer: answer.value,
      sources: sources.value,
      timestamp: new Date(),
    })
  }
}

function handleHistorySelect(item: { question: string }) {
  handleAsk(item.question)
  showHistory.value = false
}
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <div class="container mx-auto max-w-4xl px-4 py-8">
      <!-- Header -->
      <header class="mb-8 flex items-center justify-between">
        <h1 class="text-3xl font-bold text-gray-800">DropQA</h1>
        <button
          @click="showHistory = !showHistory"
          class="rounded-lg bg-gray-200 px-4 py-2 text-sm text-gray-700 hover:bg-gray-300"
        >
          {{ showHistory ? '隐藏历史' : '查看历史' }}
        </button>
      </header>

      <!-- History Panel -->
      <HistoryPanel
        v-if="showHistory"
        :history="history"
        @select="handleHistorySelect"
        @clear="clearHistory"
        class="mb-6"
      />

      <!-- Search Box -->
      <SearchBox @submit="handleAsk" :loading="loading" class="mb-6" />

      <!-- Answer Card -->
      <AnswerCard
        v-if="answer || loading || error"
        :answer="answer"
        :sources="sources"
        :loading="loading"
        :error="error"
        :mode="mode"
        :reasoning-trace="reasoningTrace"
        :progress="progress"
      />
    </div>
  </div>
</template>
