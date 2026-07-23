<script setup lang="ts">
defineProps<{ text: string; streaming: boolean; html: string | null }>();
</script>

<template>
  <div class="max-w-none">
    <!-- 完成态:一次性渲染好的 Markdown;流式态:纯文本 + 光标 -->
    <div v-if="html !== null && !streaming" class="md-body" v-html="html" />
    <div v-else class="whitespace-pre-wrap">{{ text }}<span v-if="streaming" class="cw-cursor" /></div>
  </div>
</template>
