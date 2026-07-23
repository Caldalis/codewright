/** 会话列表(侧栏数据源):REST 拉取 + 切换编排。 */

import { defineStore } from "pinia";

import type { SessionSummary } from "../lib/protocol";
import { useConversation } from "./conversation";

export const useSessions = defineStore("sessions", {
  state: () => ({
    list: [] as SessionSummary[],
    loading: false,
  }),

  actions: {
    async refresh() {
      this.loading = true;
      try {
        const resp = await fetch("/api/sessions");
        if (resp.ok) {
          this.list = ((await resp.json()) as { sessions: SessionSummary[] }).sessions;
        }
      } catch {
        // 列表拉不到不致命,侧栏显示为空即可
      } finally {
        this.loading = false;
      }
    },

    /** 切换会话;null = 新会话。 */
    async switchTo(sessionId: string | null) {
      const conversation = useConversation();
      await conversation.openSession(sessionId);
      // 新会话的 rollout 文件在引擎引导时才创建,稍等片刻再刷新列表
      setTimeout(() => void this.refresh(), 600);
    },
  },
});
