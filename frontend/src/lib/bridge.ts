/**
 * WebSocket 传输层。纯"哑管道":连接管理 + 收发帧,不含任何业务逻辑。
 * 事件通过 setEventHandler 注册的回调交给 store;依赖方向只有 store → bridge。
 *
 * 重连策略:非主动关闭时指数退避重连(0.5s 起,上限 8s),并带着同一个
 * session_id 重连 —— 会话在后端有 rollout 持久化,重连即恢复。
 */

import { ref } from "vue";

import type { BridgeEvent, ClientOp, EventMsg } from "./protocol";

export type BridgeStatus = "connecting" | "open" | "reconnecting" | "closed";

export const bridgeStatus = ref<BridgeStatus>("closed");

let socket: WebSocket | null = null;
let currentSessionId: string | null = null;
let intentionalClose = false;
let retryDelayMs = 500;
let retryTimer: ReturnType<typeof setTimeout> | null = null;

let handleEvent: (msg: EventMsg) => void = () => {};

export function setEventHandler(handler: (msg: EventMsg) => void): void {
  handleEvent = handler;
}

export function connect(sessionId: string | null): void {
  disconnect(); // 切会话:先干净地断开旧连接(不触发重连)
  currentSessionId = sessionId;
  intentionalClose = false;
  retryDelayMs = 500;
  open();
}

export function disconnect(): void {
  intentionalClose = true;
  if (retryTimer !== null) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  if (socket !== null) {
    // 先把 socket 置空再 close:close 是异步的,onclose 稍后才触发,
    // 届时靠 "socket !== ws" 识别出这是条已被抛弃的旧连接,直接忽略,
    // 不会误触发自动重连(否则切会话时旧连接的关闭会凭空拉起一条新连接)。
    const stale = socket;
    socket = null;
    stale.close();
  }
  bridgeStatus.value = "closed";
}

export function sendOp(op: ClientOp): boolean {
  if (socket === null || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(op));
  return true;
}

function open(): void {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const query = currentSessionId ? `?session_id=${encodeURIComponent(currentSessionId)}` : "";
  bridgeStatus.value = bridgeStatus.value === "closed" ? "connecting" : "reconnecting";

  const ws = new WebSocket(`${proto}//${location.host}/ws${query}`);
  socket = ws;

  ws.onopen = () => {
    bridgeStatus.value = "open";
    retryDelayMs = 500;
  };

  ws.onmessage = (frame: MessageEvent<string>) => {
    let event: BridgeEvent;
    try {
      event = JSON.parse(frame.data) as BridgeEvent;
    } catch {
      return; // 协议外的帧直接丢弃
    }
    // 首帧携带引擎分配的 session_id;记住它,重连时才能恢复同一会话
    if (event.msg.type === "session_configured") {
      currentSessionId = event.msg.session_id;
    }
    handleEvent(event.msg);
  };

  ws.onclose = (e: CloseEvent) => {
    if (socket !== ws) return; // 已被 disconnect() 抛弃的旧连接,不做任何事
    socket = null;
    if (intentionalClose) {
      bridgeStatus.value = "closed";
      return;
    }
    // 4404/4409 是桥的应用层拒绝(会话不存在/已被占用),重试也不会成功
    if (e.code === 4404 || e.code === 4409) {
      bridgeStatus.value = "closed";
      return;
    }
    bridgeStatus.value = "reconnecting";
    retryTimer = setTimeout(open, retryDelayMs);
    retryDelayMs = Math.min(retryDelayMs * 2, 8_000);
  };
}
