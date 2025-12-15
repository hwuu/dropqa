<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  loading?: boolean
}>()

const emit = defineEmits<{
  submit: [question: string]
}>()

const question = ref('')

function handleSubmit() {
  const q = question.value.trim()
  if (q && !props.loading) {
    emit('submit', q)
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSubmit()
  }
}
</script>

<template>
  <div class="flex gap-3">
    <input
      v-model="question"
      type="text"
      placeholder="请输入您的问题..."
      class="flex-1 rounded-lg border-2 border-gray-200 px-4 py-3 text-base outline-none transition-colors focus:border-blue-500"
      :disabled="loading"
      @keydown="handleKeydown"
      autofocus
    />
    <button
      @click="handleSubmit"
      :disabled="loading || !question.trim()"
      class="rounded-lg bg-blue-500 px-6 py-3 text-base font-medium text-white transition-colors hover:bg-blue-600 disabled:cursor-not-allowed disabled:bg-gray-300"
    >
      {{ loading ? '思考中...' : '提问' }}
    </button>
  </div>
</template>
