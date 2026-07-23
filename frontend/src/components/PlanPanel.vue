<script setup lang="ts">
import { Check, Circle, CircleDot } from "lucide-vue-next";

import { useConversation } from "../stores/conversation";

const conversation = useConversation();
</script>

<template>
  <aside
    v-if="conversation.plan.length"
    class="w-56 shrink-0 overflow-y-auto border-l border-edge bg-surface px-4 py-4"
  >
    <h2 class="mb-3 text-[11px] font-medium tracking-wider text-ink-faint uppercase">计划</h2>
    <ol class="space-y-2.5">
      <li
        v-for="(item, i) in conversation.plan"
        :key="i"
        class="flex items-start gap-2 text-[12.5px] leading-5"
      >
        <Check v-if="item.status === 'completed'" :size="14" class="mt-0.5 shrink-0 text-ok" />
        <CircleDot v-else-if="item.status === 'in_progress'" :size="14" class="mt-0.5 shrink-0 text-accent" />
        <Circle v-else :size="14" class="mt-0.5 shrink-0 text-ink-faint" />
        <span
          :class="{
            'text-ink-faint line-through': item.status === 'completed',
            'text-ink': item.status === 'in_progress',
            'text-ink-dim': item.status === 'pending',
          }"
        >{{ item.step }}</span>
      </li>
    </ol>
  </aside>
</template>
