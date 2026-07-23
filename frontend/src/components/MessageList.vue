<script setup lang="ts">
import { nextTick, ref, watch } from "vue";

import { useConversation } from "../stores/conversation";
import AgentMessage from "./AgentMessage.vue";
import ApprovalCard from "./ApprovalCard.vue";
import SystemNotice from "./SystemNotice.vue";
import ToolCallCard from "./ToolCallCard.vue";
import UserMessage from "./UserMessage.vue";

const conversation = useConversation();
const scroller = ref<HTMLElement | null>(null);

// 跟随滚动:只有当用户本来就在底部附近时才自动滚,翻旧消息时不打扰
watch(
  () => [conversation.blocks.length, lastBlockSize()],
  async () => {
    const el = scroller.value;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (!nearBottom) return;
    await nextTick();
    el.scrollTop = el.scrollHeight;
  },
);

function lastBlockSize(): number {
  const last = conversation.blocks[conversation.blocks.length - 1];
  if (!last) return 0;
  if (last.kind === "agent") return last.text.length;
  if (last.kind === "tool") return last.body.length + (last.status === "running" ? 0 : 1);
  return 1;
}
</script>

<template>
  <div ref="scroller" class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto max-w-3xl space-y-4 px-6 py-6">
      <template v-if="conversation.blocks.length">
        <template v-for="block in conversation.blocks" :key="block.id">
          <UserMessage v-if="block.kind === 'user'" :text="block.text" />
          <AgentMessage
            v-else-if="block.kind === 'agent'"
            :text="block.text"
            :streaming="block.streaming"
            :html="block.html"
          />
          <ToolCallCard v-else-if="block.kind === 'tool'" :block="block" />
          <ApprovalCard v-else-if="block.kind === 'approval'" :block="block" />
          <SystemNotice v-else :level="block.level" :text="block.text" />
        </template>
      </template>

      <!-- 空状态 -->
      <div v-else class="flex h-[60vh] flex-col items-center justify-center gap-2">
        <p class="text-[15px] text-ink-dim">开始一个任务</p>
        <p class="text-[12.5px] text-ink-faint">
          描述你想改什么、查什么、跑什么,回车发送
        </p>
      </div>
    </div>
  </div>
</template>
