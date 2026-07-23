/**
 * 对话状态:整个 UI 唯一的事件消费者。
 * bridge 收到的每一帧都进 onEvent,由这里决定"追加块 / 更新块 / 改全局状态"。
 * 组件只读本 store 渲染,不直接碰 WebSocket。
 */

import { defineStore } from "pinia";

import * as bridge from "../lib/bridge";
import { renderMarkdown } from "../lib/markdown";
import type {
  EventMsg,
  HistoryBlock,
  PlanItem,
  ReviewDecision,
} from "../lib/protocol";

export type Block =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "agent"; text: string; streaming: boolean; html: string | null }
  | {
      id: number;
      kind: "tool";
      callId: string;
      name: string;
      args: Record<string, unknown>;
      status: "running" | "ok" | "fail";
      body: string;
      startedAt: number;
      elapsedMs: number | null;
    }
  | {
      id: number;
      kind: "approval";
      requestId: string;
      opType: "exec_approval_response" | "patch_approval_response";
      summary: string;
      decision: ReviewDecision | null;
    }
  | { id: number; kind: "notice"; level: "info" | "warn" | "error"; text: string };

let nextBlockId = 1;

export const useConversation = defineStore("conversation", {
  state: () => ({
    blocks: [] as Block[],
    turnRunning: false,
    sessionId: null as string | null,
    model: "",
    cwd: "",
    plan: [] as PlanItem[],
    tokens: { input: 0, output: 0, total: 0 },
  }),

  getters: {
    title(state): string {
      const firstUser = state.blocks.find((b) => b.kind === "user");
      if (firstUser?.kind === "user") {
        const line = firstUser.text.split("\n")[0] ?? "";
        return line.length > 40 ? line.slice(0, 39) + "…" : line;
      }
      return "新会话";
    },
    pendingApproval(state): Extract<Block, { kind: "approval" }> | null {
      for (let i = state.blocks.length - 1; i >= 0; i--) {
        const b = state.blocks[i];
        if (b?.kind === "approval" && b.decision === null) return b;
      }
      return null;
    },
  },

  actions: {
    /** 切到某个会话(null = 新会话)。先清空、播历史,再接实时流。 */
    async openSession(sessionId: string | null) {
      this.$reset();
      nextBlockId = 1;
      if (sessionId) {
        try {
          const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/history`);
          if (resp.ok) {
            const data = (await resp.json()) as { blocks: HistoryBlock[] };
            for (const h of data.blocks) this.seedHistoryBlock(h);
          }
        } catch {
          this.pushNotice("warn", "历史记录加载失败,仅接入实时流");
        }
      }
      bridge.setEventHandler((msg) => this.onEvent(msg));
      bridge.connect(sessionId);
    },

    seedHistoryBlock(h: HistoryBlock) {
      if (h.kind === "user") {
        this.blocks.push({ id: nextBlockId++, kind: "user", text: h.text });
      } else if (h.kind === "agent") {
        const block: Block = {
          id: nextBlockId++, kind: "agent", text: h.text, streaming: false, html: null,
        };
        this.blocks.push(block);
        void this.renderAgentHtml(block.id, h.text);
      } else {
        this.blocks.push({
          id: nextBlockId++, kind: "tool", callId: `history-${nextBlockId}`,
          name: h.name, args: {}, status: h.success ? "ok" : "fail",
          body: h.body, startedAt: 0, elapsedMs: null,
        });
      }
    },

    /** 事件 → UI 的完整映射表 */
    onEvent(msg: EventMsg) {
      switch (msg.type) {
        case "session_configured":
          this.sessionId = msg.session_id;
          this.model = msg.model;
          this.cwd = msg.cwd;
          break;
        case "turn_started":
          this.turnRunning = true;
          break;
        case "turn_completed":
          this.turnRunning = false;
          this.finishStreaming();
          break;
        case "turn_aborted":
          this.turnRunning = false;
          this.finishStreaming();
          this.pushNotice(
            msg.reason === "interrupted" ? "info" : "error",
            msg.reason === "interrupted" ? "已中断" : "本轮因错误终止",
          );
          break;
        case "agent_message_delta":
          this.appendDelta(msg.delta);
          break;
        case "agent_message": {
          // 完整消息到达:替换流式积累的文本,并触发一次性 Markdown 渲染
          const block = this.currentStreamingBlock() ?? this.newAgentBlock();
          block.text = msg.content;
          block.streaming = false;
          void this.renderAgentHtml(block.id, msg.content);
          break;
        }
        case "agent_reasoning":
          break; // MVP 不展示推理过程
        case "tool_call_started":
          this.finishStreaming();
          this.blocks.push({
            id: nextBlockId++, kind: "tool", callId: msg.call_id, name: msg.tool_name,
            args: msg.arguments, status: "running", body: "", startedAt: Date.now(),
            elapsedMs: null,
          });
          break;
        case "tool_call_completed": {
          const tool = this.findToolBlock(msg.call_id);
          if (tool) {
            tool.status = msg.success ? "ok" : "fail";
            tool.body = msg.body;
            tool.elapsedMs = tool.startedAt > 0 ? Date.now() - tool.startedAt : null;
          }
          break;
        }
        case "exec_approval_request":
        case "patch_approval_request":
          this.blocks.push({
            id: nextBlockId++, kind: "approval", requestId: msg.request_id,
            opType: msg.type === "exec_approval_request"
              ? "exec_approval_response" : "patch_approval_response",
            summary: msg.action.summary, decision: null,
          });
          break;
        case "plan_update":
          this.plan = msg.plan;
          break;
        case "token_count":
          this.tokens = { input: msg.input, output: msg.output, total: msg.total };
          break;
        case "compaction_started":
          this.pushNotice("info", "上下文接近上限,正在压缩历史……");
          break;
        case "compaction_completed":
          this.pushNotice(
            "info",
            `历史已压缩:${formatK(msg.tokens_before)} → ${formatK(msg.tokens_after)} tokens`,
          );
          break;
        case "warning":
          this.pushNotice("warn", msg.message);
          break;
        case "error":
        case "bridge_error":
          this.pushNotice("error", msg.message);
          break;
        case "shutdown_complete":
          break;
      }
    },

    sendUserMessage(text: string): boolean {
      const trimmed = text.trim();
      if (!trimmed) return false;
      const ok = bridge.sendOp({
        type: "user_turn",
        items: [{ type: "text", text: trimmed }],
      });
      if (ok) this.blocks.push({ id: nextBlockId++, kind: "user", text: trimmed });
      return ok;
    },

    interrupt() {
      bridge.sendOp({ type: "interrupt" });
    },

    decideApproval(block: Extract<Block, { kind: "approval" }>, decision: ReviewDecision) {
      const ok = bridge.sendOp({
        type: block.opType,
        request_id: block.requestId,
        decision,
      });
      if (ok) block.decision = decision;
    },

    // ── 内部辅助 ──────────────────────────────────────────────

    currentStreamingBlock(): Extract<Block, { kind: "agent" }> | null {
      const last = this.blocks[this.blocks.length - 1];
      return last?.kind === "agent" && last.streaming ? last : null;
    },

    newAgentBlock(): Extract<Block, { kind: "agent" }> {
      const block: Block = {
        id: nextBlockId++, kind: "agent", text: "", streaming: true, html: null,
      };
      this.blocks.push(block);
      return block as Extract<Block, { kind: "agent" }>;
    },

    appendDelta(delta: string) {
      const block = this.currentStreamingBlock() ?? this.newAgentBlock();
      block.text += delta;
    },

    finishStreaming() {
      const block = this.currentStreamingBlock();
      if (block) {
        block.streaming = false;
        void this.renderAgentHtml(block.id, block.text);
      }
    },

    async renderAgentHtml(blockId: number, text: string) {
      const html = await renderMarkdown(text);
      const block = this.blocks.find((b) => b.id === blockId);
      if (block?.kind === "agent") block.html = html;
    },

    findToolBlock(callId: string): Extract<Block, { kind: "tool" }> | null {
      for (let i = this.blocks.length - 1; i >= 0; i--) {
        const b = this.blocks[i];
        if (b?.kind === "tool" && b.callId === callId) return b;
      }
      return null;
    },

    pushNotice(level: "info" | "warn" | "error", text: string) {
      this.blocks.push({ id: nextBlockId++, kind: "notice", level, text });
    },
  },
});

function formatK(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}
