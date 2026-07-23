<script setup lang="ts">
import { computed } from "vue";

import { bridgeStatus } from "../lib/bridge";
import { formatTokens } from "../lib/format";
import { useConversation } from "../stores/conversation";

const conversation = useConversation();

const statusInfo = computed(() => {
  switch (bridgeStatus.value) {
    case "open": return { color: "bg-ok", label: "已连接" };
    case "connecting": return { color: "bg-warn", label: "连接中" };
    case "reconnecting": return { color: "bg-warn", label: "重连中" };
    default: return { color: "bg-danger", label: "已断开" };
  }
});
</script>

<template>
  <header class="flex h-12 shrink-0 items-center justify-between border-b border-edge px-4">
    <h1 class="truncate text-[13px] font-medium text-ink-dim">{{ conversation.title }}</h1>

    <div class="flex items-center gap-4 text-[12px] text-ink-faint">
      <span v-if="conversation.tokens.total" class="font-mono">
        {{ formatTokens(conversation.tokens.total) }} tok
      </span>
      <span v-if="conversation.model" class="font-mono">{{ conversation.model }}</span>
      <span class="flex items-center gap-1.5">
        <span class="size-1.5 rounded-full" :class="statusInfo.color" />
        {{ statusInfo.label }}
      </span>
    </div>
  </header>
</template>
