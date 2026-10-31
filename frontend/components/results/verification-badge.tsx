import { AlertCircle, CheckCircle2, Clock3, MinusCircle, XCircle } from "lucide-react";
import type { TraceStatus, VerificationStatus } from "@/types/vera";
import { verificationLabel } from "@/lib/result-utils";

export function VerificationBadge({ status }: { status: VerificationStatus }) {
  const style = status === "verified" ? "bg-[#e9f5ed] text-[#207351] border-[#cce8d7]" :
    status === "rejected" ? "bg-[#fff0ed] text-[#a3443c] border-[#f0d2cc]" :
    status === "needs_more_analysis" ? "bg-[#fff6e8] text-[#976827] border-[#f0e0c3]" :
    "bg-[#eef1ef] text-[#718076] border-[#e0e6e0]";
  const Icon = status === "verified" ? CheckCircle2 : status === "rejected" ? XCircle : status === "needs_more_analysis" ? AlertCircle : MinusCircle;
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold ${style}`}>
    <Icon size={12} aria-hidden="true" />{verificationLabel(status)}
  </span>;
}

export function ExecutionStatus({ status }: { status: TraceStatus }) {
  const failed = status === "failed";
  return <span className={`inline-flex items-center gap-1 text-[11px] font-medium ${failed ? "text-[#ad4b41]" : "text-[#74857a]"}`}>
    {failed ? <XCircle size={12} /> : status === "completed" ? <CheckCircle2 size={12} /> : <Clock3 size={12} />}
    {failed ? "Execution failed" : status === "completed" ? "Completed" : status.replaceAll("_", " ")}
  </span>;
}
