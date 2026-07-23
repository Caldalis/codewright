<script setup lang="ts">
import { ArrowUp, Square } from "lucide-vue-next";
import { computed, ref } from "vue";

import { bridgeStatus } from "../lib/bridge";
import { useConversation } from "../stores/conversation";

const conversation = useConversation();
const draft = ref("");
const textareaEl = ref<HTMLTextAreaElement | null>(null);

const canSend = computed(
  () => bridgeStatus.value === "open" && draft.value.trim().length > 0,
);

function send() {
  if (!canSend.value) return;
  if (conversation.sendUserMessage(draft.value)) {
    draft.value = "";
    resize();
  }
}

function onKeydown(e: KeyboardEvent) {
  // Enter 发送,Shift+Enter 换行;输入法组词中的 Enter 不触发发送
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
}

function resize() {
  const el = textareaEl.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
}
</script>

<template>
  <div class="shrink-0 px-6 pb-5">
    <div class="mx-auto max-w-3xl">
      <div
        class="flex items-end gap-2 rounded-xl border border-edge bg-surface px-3 py-2.5 transition-colors focus-within:border-edge-strong"
      >
        <textarea
          ref="textareaEl"
          v-model="draft"
          rows="1"
          placeholder="描述任务…(Enter 发送,Shift+Enter 换行)"
          class="max-h-50 flex-1 resize-none bg-transparent text-[14px] leading-6 outline-none placeholder:text-ink-faint"
          @keydown="onKeydown"
          @input="resize"
        />

        <!-- 运行中显示"停止",否则显示"发送" -->
        <button
          v-if="conversation.turnRunning"
          class="grid size-8 shrink-0 place-items-center rounded-lg bg-raised text-ink-dim transition-colors hover:bg-hover hover:text-danger"
          title="中断当前任务"
          @click="conversation.interrupt()"
        >
          <Square :size="13" fill="currentColor" />
        </button>
        <button
          v-else
          class="grid size-8 shrink-0 place-items-center rounded-lg transition-colors"
          :class="canSend ? 'bg-accent text-bg hover:opacity-90' : 'bg-raised text-ink-faint'"
          :disabled="!canSend"
          title="发送"
          @click="send"
        >
          <ArrowUp :size="16" />
        </button>
      </div>

      <p v-if="bridgeStatus !== 'open'" class="mt-1.5 text-center text-[11.5px] text-warn/80">
        连接已断开,正在重连……期间无法发送消息
      </p>
    </div>
  </div>
</template>
