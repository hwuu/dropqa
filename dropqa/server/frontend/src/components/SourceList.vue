<script setup lang="ts">
import type { Source } from '@/types'

defineProps<{
  sources: Source[]
}>()
</script>

<template>
  <div class="border-t border-gray-100 pt-4">
    <h3 class="mb-3 text-sm font-medium text-gray-500">引用来源</h3>
    <div class="space-y-2">
      <div
        v-for="(source, index) in sources"
        :key="index"
        class="flex gap-3 rounded-lg bg-gray-50 p-3"
      >
        <!-- Number Badge -->
        <div
          class="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-blue-500 text-xs font-bold text-white"
        >
          {{ index + 1 }}
        </div>

        <!-- Content -->
        <div class="min-w-0 flex-1">
          <div class="mb-1 text-xs text-blue-600">
            {{ source.document_name }}
            <span v-if="source.path" class="text-gray-400"> &gt; {{ source.path }}</span>
          </div>
          <div class="line-clamp-2 text-sm text-gray-600">
            {{ source.content_snippet }}
          </div>
        </div>

        <!-- Score Badge (if available) -->
        <div
          v-if="source.score !== undefined"
          class="flex-shrink-0 text-xs text-gray-400"
          :title="`相关度分数: ${source.score.toFixed(3)}`"
        >
          {{ (source.score * 100).toFixed(0) }}%
        </div>
      </div>
    </div>
  </div>
</template>
