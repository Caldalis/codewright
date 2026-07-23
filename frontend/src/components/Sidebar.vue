<script setup lang="ts">
import { PanelLeftClose, PanelLeftOpen, Plus } from "lucide-vue-next";
import { computed, onMounted, ref, watch } from "vue";

import { relativeTime } from "../lib/format";
import { useConversation } from "../stores/conversation";
import { useSessions } from "../stores/sessions";
import BrandMark from "./BrandMark.vue";

const sessions = useSessions();
const conversation = useConversation();

const collapsed = ref(localStorage.getItem("cw.sidebar.collapsed") === "1");

function toggle() {
  collapsed.value = !collapsed.value;
  localStorage.setItem("cw.sidebar.collapsed", collapsed.value ? "1" : "0");
}

// 没有标题 = 从未真正用过的空会话,列表里不显示
const visibleSessions = computed(() => sessions.list.filter((s) => s.title !== ""));

function isCurrent(sessionId: string): boolean {
  return conversation.sessionId === sessionId;
}

async function open(sessionId: string) {
  if (isCurrent(sessionId)) return;
  await sessions.switchTo(sessionId);
}

onMounted(() => void sessions.refresh());

// 每轮任务结束后刷新列表:当前会话的标题(取自首条用户消息)此时才落盘
watch(
  () => conversation.turnRunning,
  (running, wasRunning) => {
    if (wasRunning && !running) void sessions.refresh();
  },
);
</script>

<template>
  <aside
    class="flex h-full shrink-0 flex-col border-r border-edge bg-surface transition-[width] duration-200"
    :class="collapsed ? 'w-[52px]' : 'w-60'"
  >
    <div class="flex items-center" :class="collapsed ? 'flex-col' : 'justify-between pr-2'">
      <BrandMark :collapsed="collapsed" />
      <button
        class="grid size-7 place-items-center rounded-md text-ink-faint hover:bg-hover hover:text-ink-dim"
        :title="collapsed ? '展开侧栏' : '收起侧栏'"
        @click="toggle"
      >
        <PanelLeftOpen v-if="collapsed" :size="15" />
        <PanelLeftClose v-else :size="15" />
      </button>
    </div>

    <div class="px-2 pt-2">
      <button
        class="flex w-full items-center gap-2 rounded-lg border border-edge px-2.5 py-1.5 text-[13px] text-ink-dim transition-colors hover:border-edge-strong hover:bg-hover hover:text-ink"
        :class="collapsed && 'justify-center px-0'"
        title="新会话"
        @click="sessions.switchTo(null)"
      >
        <Plus :size="15" class="shrink-0" />
        <span v-if="!collapsed">新会话</span>
      </button>
    </div>

    <nav v-if="!collapsed" class="mt-3 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
      <button
        v-for="s in visibleSessions"
        :key="s.session_id"
        class="relative block w-full rounded-lg px-2.5 py-2 text-left transition-colors"
        :class="isCurrent(s.session_id) ? 'bg-raised' : 'hover:bg-hover'"
        @click="open(s.session_id)"
      >
        <span
          v-if="isCurrent(s.session_id)"
          class="absolute top-2 bottom-2 left-0 w-0.5 rounded-full bg-accent"
        />
        <span class="block truncate text-[13px] leading-5" :class="isCurrent(s.session_id) ? 'text-ink' : 'text-ink-dim'">
          {{ s.title }}
        </span>
        <span class="block text-[11px] text-ink-faint">{{ relativeTime(s.start_time) }}</span>
      </button>

      <p v-if="!visibleSessions.length && !sessions.loading" class="px-2.5 pt-2 text-[12px] text-ink-faint">
        暂无历史会话
      </p>
    </nav>
    <div v-else class="flex-1" />
  </aside>
</template>
