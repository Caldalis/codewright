<script setup lang="ts">
defineProps<{ text: string }>();
</script>

<template>
  <div class="flex justify-end">
    <div class="max-w-[85%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 whitespace-pre-wrap">
      {{ text }}
    </div>
  </div>
</template>
