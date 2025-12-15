<script setup lang="ts">
import type { HistoryItem } from '@/types'

defineProps<{
  history: HistoryItem[]
}>()

const emit = defineEmits<{
  select: [item: HistoryItem]
  clear: []
}>()

function formatTime(date: Date): string {
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  // Less than 1 minute
  if (diff < 60 * 1000) {
    return '刚刚'
  }

  // Less than 1 hour
  if (diff < 60 * 60 * 1000) {
    return `${Math.floor(diff / (60 * 1000))} 分钟前`
  }

  // Less than 1 day
  if (diff < 24 * 60 * 60 * 1000) {
    return `${Math.floor(diff / (60 * 60 * 1000))} 小时前`
  }

  // Format as date
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<template>
  <div class="rounded-xl bg-white p-4 shadow-md">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="font-medium text-gray-700">历史记录</h2>
      <button
        v-if="history.length > 0"
        @click="emit('clear')"
        class="text-xs text-red-500 hover:text-red-700"
      >
        清空历史
      </button>
    </div>

    <div v-if="history.length === 0" class="py-4 text-center text-sm text-gray-400">
      暂无历史记录
    </div>

    <div v-else class="max-h-64 space-y-2 overflow-y-auto">
      <button
        v-for="(item, index) in history"
        :key="index"
        @click="emit('select', item)"
        class="w-full rounded-lg bg-gray-50 p-3 text-left transition-colors hover:bg-gray-100"
      >
        <div class="flex items-start justify-between gap-2">
          <span class="line-clamp-1 flex-1 text-sm text-gray-700">{{ item.question }}</span>
          <span class="flex-shrink-0 text-xs text-gray-400">{{ formatTime(item.timestamp) }}</span>
        </div>
        <div class="mt-1 line-clamp-1 text-xs text-gray-500">
          {{ item.answer.substring(0, 100) }}...
        </div>
      </button>
    </div>
  </div>
</template>
