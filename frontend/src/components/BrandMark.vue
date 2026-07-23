<script setup lang="ts">
import { ArrowUpRight } from "lucide-vue-next";

defineProps<{ collapsed: boolean }>();
</script>

<template>
  <a
    href="https://github.com/Caldalis/codewright"
    target="_blank"
    rel="noopener"
    class="group flex h-12 items-center gap-2.5 px-3 select-none"
    title="github.com/Caldalis/codewright"
  >
    <!-- 图标:cw 双字母 + 底部刻痕(wright = 工匠,凿痕是它的语义) -->
    <svg width="28" height="28" viewBox="0 0 28 28" class="shrink-0">
      <rect x="0.5" y="0.5" width="27" height="27" rx="7" fill="#1c1c21" stroke="rgba(255,255,255,0.14)" />
      <text
        x="14" y="16.5"
        text-anchor="middle"
        font-family="JetBrains Mono, monospace"
        font-size="11.5" font-weight="600"
        letter-spacing="-0.5"
        fill="#55b7f0"
      >cw</text>
      <rect x="8" y="20" width="12" height="2" rx="1" fill="#55b7f0" opacity="0.85" />
    </svg>

    <template v-if="!collapsed">
      <span class="text-[15px] font-medium tracking-tight text-ink/90 group-hover:text-ink">
        codewright
      </span>
      <ArrowUpRight
        :size="13"
        class="text-ink-faint opacity-0 transition-opacity duration-150 group-hover:opacity-100"
      />
    </template>
  </a>
</template>
