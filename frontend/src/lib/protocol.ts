/**
 * 与后端 protocol/(op.py / event.py)逐字段对齐的 TS 类型。
 * 后端是带 "type" 判别字段的 pydantic tagged union,这里用可辨识联合一一映射。
 */

export interface PendingAction {
  action_id: string;
  kind: "exec" | "patch" | "network";
  summary: string;
  details: Record<string, unknown>;
}

export interface PlanItem {
  step: string;
  status: "pending" | "in_progress" | "completed";
}

/** 引擎 → 浏览器(Event.msg),外加桥自己的 bridge_error */
export type EventMsg =
  | { type: "session_configured"; session_id: string; model: string; cwd: string; permission_profile: string }
  | { type: "turn_started"; turn_id: string }
  | { type: "turn_completed"; turn_id: string; last_agent_message: string | null }
  | { type: "turn_aborted"; turn_id: string; reason: "interrupted" | "error" }
  | { type: "agent_message"; content: string }
  | { type: "agent_message_delta"; delta: string }
  | { type: "agent_reasoning"; content: string }
  | { type: "tool_call_started"; call_id: string; tool_name: string; arguments: Record<string, unknown> }
  | { type: "tool_call_completed"; call_id: string; success: boolean; body: string }
  | { type: "exec_approval_request"; request_id: string; action: PendingAction }
  | { type: "patch_approval_request"; request_id: string; action: PendingAction }
  | { type: "plan_update"; plan: PlanItem[]; explanation: string | null }
  | { type: "token_count"; input: number; output: number; total: number }
  | { type: "compaction_started"; reason: string; tokens_before: number }
  | { type: "compaction_completed"; tokens_before: number; tokens_after: number }
  | { type: "error"; message: string }
  | { type: "warning"; message: string }
  | { type: "shutdown_complete" }
  | { type: "bridge_error"; message: string };

export interface BridgeEvent {
  id: string;
  msg: EventMsg;
}

/** 浏览器 → 引擎(桥的白名单内) */
export type ClientOp =
  | { type: "user_turn"; items: Array<{ type: "text"; text: string }> }
  | { type: "interrupt" }
  | { type: "exec_approval_response"; request_id: string; decision: ReviewDecision }
  | { type: "patch_approval_response"; request_id: string; decision: ReviewDecision };

export type ReviewDecision = "approved" | "approved_for_session" | "denied" | "abort";

/** REST: GET /api/sessions */
export interface SessionSummary {
  session_id: string;
  title: string;
  model: string;
  start_time: number;
  active: boolean;
}

/** REST: GET /api/sessions/{id}/history */
export type HistoryBlock =
  | { kind: "user"; text: string }
  | { kind: "agent"; text: string }
  | { kind: "tool"; name: string; success: boolean; body: string };
