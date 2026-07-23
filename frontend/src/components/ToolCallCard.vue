<script setup lang="ts">
import {
  BookOpen,
  Check,
  ChevronRight,
  FileDiff,
  FileText,
  Folder,
  FolderSearch,
  GitBranch,
  ListChecks,
  Loader,
  Search,
  SquareTerminal,
  Wrench,
  X,
} from "lucide-vue-next";
import { computed, ref, type Component } from "vue";

import { summarizeArgs } from "../lib/format";
import type { Block } from "../stores/conversation";

const props = defineProps<{ block: Extract<Block, { kind: "tool" }> }>();

const expanded = ref(false);

const TOOL_ICONS: Record<string, Component> = {
  shell: SquareTerminal,
  shell_output: SquareTerminal,
  shell_kill: SquareTerminal,
  apply_patch: FileDiff,
  read_file: FileText,
  list_dir: Folder,
  find_files: FolderSearch,
  search_text: Search,
  update_plan: ListChecks,
  skill: BookOpen,
  spawn_agent: GitBranch,
  send_message: GitBranch,
  followup_task: GitBranch,
  wait_agent: GitBranch,
  close_agent: GitBranch,
  list_agents: GitBranch,
};

const icon = computed(() => TOOL_ICONS[props.block.name] ?? Wrench);
const argSummary = computed(() => summarizeArgs(props.block.args));
const elapsed = computed(() => {
  if (props.block.elapsedMs === null) return "";
  const s = props.block.elapsedMs / 1000;
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
});
</script>

<template>
  <div class="overflow-hidden rounded-lg border border-edge bg-surface">
    <!-- 头部一行:默认收起,点击展开输出 -->
    <button
      class="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-hover"
      @click="expanded = !expanded"
    >
      <ChevronRight
        :size="14"
        class="shrink-0 text-ink-faint transition-transform duration-150"
        :class="expanded && 'rotate-90'"
      />
      <component :is="icon" :size="14" class="shrink-0 text-ink-dim" />
      <span class="shrink-0 font-mono text-[12.5px] text-ink-dim">{{ block.name }}</span>
      <span v-if="argSummary" class="min-w-0 flex-1 truncate font-mono text-[12.5px] text-ink-faint">
        {{ argSummary }}
      </span>
      <span v-else class="flex-1" />
      <span v-if="elapsed" class="shrink-0 font-mono text-[11px] text-ink-faint">{{ elapsed }}</span>
      <Loader v-if="block.status === 'running'" :size="14" class="shrink-0 animate-spin text-accent" />
      <Check v-else-if="block.status === 'ok'" :size="14" class="shrink-0 text-ok" />
      <X v-else :size="14" class="shrink-0 text-danger" />
    </button>

    <div v-if="expanded && block.body" class="border-t border-edge">
      <pre class="max-h-80 overflow-auto px-3 py-2.5 font-mono text-[12px] leading-relaxed whitespace-pre-wrap text-ink-dim">{{ block.body }}</pre>
    </div>
  </div>
</template>
