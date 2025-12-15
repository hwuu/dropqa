<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import type { Source, ReasoningStep } from '@/types'
import SourceList from './SourceList.vue'
import CopyButton from './CopyButton.vue'
import ReasoningTrace from './ReasoningTrace.vue'

const props = defineProps<{
  answer: string
  sources: Source[]
  loading: boolean
  error: string
  mode: 'simple' | 'agentic'
  reasoningTrace?: ReasoningStep[]
  progress?: string
}>()

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true,
})

const renderedAnswer = computed(() => {
  if (!props.answer) return ''
  return marked.parse(props.answer) as string
})

const progressText = computed(() => {
  return props.progress || '正在思考...'
})
</script>

<template>
  <div class="rounded-xl bg-white p-6 shadow-md">
    <!-- Loading State -->
    <div v-if="loading" class="flex items-center justify-center py-8 text-gray-500">
      <svg class="mr-3 h-5 w-5 animate-spin" viewBox="0 0 24 24">
        <circle
          class="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          stroke-width="4"
          fill="none"
        />
        <path
          class="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      <span>{{ progressText }}</span>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="rounded-lg bg-red-50 p-4 text-red-600">
      {{ error }}
    </div>

    <!-- Answer Content -->
    <template v-else-if="answer">
      <!-- Mode Badge -->
      <div class="mb-4 flex items-center justify-between">
        <span
          v-if="mode === 'agentic'"
          class="rounded-full bg-purple-100 px-3 py-1 text-xs font-medium text-purple-700"
        >
          Agentic RAG
        </span>
        <span v-else class="rounded-full bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700">
          Simple RAG
        </span>
        <CopyButton :text="answer" />
      </div>

      <!-- Reasoning Trace (for Agentic mode) -->
      <ReasoningTrace
        v-if="mode === 'agentic' && reasoningTrace && reasoningTrace.length > 0"
        :steps="reasoningTrace"
        class="mb-4"
      />

      <!-- Answer -->
      <div class="prose max-w-none" v-html="renderedAnswer" />

      <!-- Sources -->
      <SourceList v-if="sources.length > 0" :sources="sources" class="mt-6" />
    </template>
  </div>
</template>
