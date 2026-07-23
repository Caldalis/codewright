<script setup lang="ts">
import { ShieldAlert } from "lucide-vue-next";

import type { ReviewDecision } from "../lib/protocol";
import { useConversation, type Block } from "../stores/conversation";

const props = defineProps<{ block: Extract<Block, { kind: "approval" }> }>();
const conversation = useConversation();

const DECISION_LABELS: Record<ReviewDecision, string> = {
  approved: "允许",
  approved_for_session: "本会话允许",
  denied: "拒绝",
  abort: "中止任务",
};

function decide(decision: ReviewDecision) {
  conversation.decideApproval(props.block, decision);
}
</script>

<template>
  <div class="rounded-lg border border-warn/25 bg-warn/[0.04] px-3.5 py-3">
    <div class="flex items-start gap-2.5">
      <ShieldAlert :size="16" class="mt-0.5 shrink-0 text-warn" />
      <div class="min-w-0 flex-1">
        <p class="text-[12px] font-medium text-warn">需要审批</p>
        <p class="mt-1 font-mono text-[12.5px] break-all text-ink-dim">{{ block.summary }}</p>

        <!-- 未决:四个决定按钮;已决:只留结果一行 -->
        <div v-if="block.decision === null" class="mt-3 flex flex-wrap gap-2">
          <button
            class="rounded-md bg-raised px-3 py-1 text-[12.5px] text-ink transition-colors hover:bg-hover"
            @click="decide('approved')"
          >允许</button>
          <button
            class="rounded-md bg-raised px-3 py-1 text-[12.5px] text-ink-dim transition-colors hover:bg-hover hover:text-ink"
            @click="decide('approved_for_session')"
          >本会话允许</button>
          <button
            class="rounded-md px-3 py-1 text-[12.5px] text-danger/90 transition-colors hover:bg-danger/10"
            @click="decide('denied')"
          >拒绝</button>
          <button
            class="rounded-md px-3 py-1 text-[12.5px] text-danger/70 transition-colors hover:bg-danger/10"
            @click="decide('abort')"
          >中止任务</button>
        </div>
        <p v-else class="mt-2 text-[12px] text-ink-faint">
          已选择:{{ DECISION_LABELS[block.decision] }}
        </p>
      </div>
    </div>
  </div>
</template>
