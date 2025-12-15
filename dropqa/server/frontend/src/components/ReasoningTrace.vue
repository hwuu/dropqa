<script setup lang="ts">
import { ref } from 'vue'
import type { ReasoningStep } from '@/types'

defineProps<{
  steps: ReasoningStep[]
}>()

const expanded = ref(false)
</script>

<template>
  <div class="rounded-lg border border-purple-200 bg-purple-50">
    <!-- Header -->
    <button
      @click="expanded = !expanded"
      class="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-purple-700 hover:bg-purple-100"
    >
      <span class="flex items-center gap-2">
        <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
          />
        </svg>
        推理过程 ({{ steps.length }} 步)
      </span>
      <svg
        class="h-4 w-4 transition-transform"
        :class="{ 'rotate-180': expanded }"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
      </svg>
    </button>

    <!-- Content -->
    <div v-if="expanded" class="border-t border-purple-200 px-4 py-3">
      <div class="space-y-3">
        <div v-for="(step, index) in steps" :key="index" class="flex gap-3">
          <!-- Step Number -->
          <div
            class="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-purple-200 text-xs font-bold text-purple-700"
          >
            {{ index + 1 }}
          </div>

          <!-- Step Content -->
          <div class="min-w-0 flex-1">
            <div class="text-sm font-medium text-purple-800">{{ step.step }}</div>
            <div class="mt-1 text-xs text-purple-600">{{ step.action }}</div>
            <div
              v-if="step.result"
              class="mt-1 line-clamp-2 rounded bg-white p-2 text-xs text-gray-600"
            >
              {{ step.result }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
