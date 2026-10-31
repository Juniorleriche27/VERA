"use client";

import { RefreshCw } from "lucide-react";

type ApiState = "checking" | "connected" | "unavailable";

export function ApiStatus({ state, onRetry }: { state: ApiState; onRetry: () => void }) {
  const label = state === "connected" ? "Connected" : state === "checking" ? "Checking API" : "Unavailable";
  return <div className="flex items-center gap-2 text-xs font-semibold text-[#64746c]" aria-live="polite">
    <span className={`status-dot ${state === "connected" ? "status-dot--on" : state === "checking" ? "status-dot--busy" : "status-dot--off"}`} aria-hidden="true" />
    <span className="hidden sm:inline">API</span><span>{label}</span>
    {state === "unavailable" && <button type="button" onClick={onRetry} aria-label="Retry API connection" title="Retry API connection" className="ml-1 rounded-md p-1.5 hover:bg-[#e9eeea]"><RefreshCw size={14} /></button>}
  </div>;
}
