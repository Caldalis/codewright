/** 小工具:相对时间、token 数、工具参数摘要。 */

export function relativeTime(epochSeconds: number): string {
  const diffMs = Date.now() - epochSeconds * 1000;
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} 天前`;
  const d = new Date(epochSeconds * 1000);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function formatTokens(total: number): string {
  if (total < 1000) return String(total);
  return `${(total / 1000).toFixed(1)}k`;
}

/** 从工具参数里挑最有信息量的一个值做单行摘要(命令、路径、模式……) */
export function summarizeArgs(args: Record<string, unknown>): string {
  for (const key of ["command", "path", "pattern", "query", "target", "name", "message"]) {
    const value = args[key];
    if (typeof value === "string" && value.trim()) {
      const first = value.trim().split("\n")[0] ?? "";
      return first.length > 70 ? first.slice(0, 69) + "…" : first;
    }
  }
  return "";
}
