<script setup lang="ts">
defineProps<{ level: "info" | "warn" | "error"; text: string }>();
</script>

<template>
  <p
    class="py-0.5 text-center text-[12px]"
    :class="{
      'text-ink-faint': level === 'info',
      'text-warn/80': level === 'warn',
      'text-danger/90': level === 'error',
    }"
  >
    {{ text }}
  </p>
</template>
