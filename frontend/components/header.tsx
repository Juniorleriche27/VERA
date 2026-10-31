import { Activity, ArrowUpRight } from "lucide-react";
import { ApiStatus } from "@/components/api-status";

export function Header({ apiState, retry }: { apiState: "checking" | "connected" | "unavailable"; retry: () => void }) {
  return <header className="border-b border-[#dce3dc] bg-white/90">
    <div className="mx-auto flex h-[76px] max-w-[1320px] items-center justify-between gap-4 px-5 sm:px-8 lg:px-12">
      <div className="flex items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-xl bg-[#176b57] text-white"><Activity size={21} strokeWidth={2.1}/></div>
        <div className="flex flex-col"><span className="text-[21px] font-black tracking-[-.07em] leading-[1.1]">VERA</span><span className="mt-1 text-[10px] font-semibold tracking-[.04em] text-[#7a8580]">VERIFIABLE ANALYTICS AGENT</span></div>
      </div>
      <div className="flex items-center gap-5 sm:gap-8">
        <span className="hidden items-center gap-1.5 text-[11px] font-semibold text-[#8c9790] lg:flex">INVESTIGATION WORKSPACE <ArrowUpRight size={13}/></span>
        <span className="h-5 w-px bg-[#e0e5df]" aria-hidden="true" />
        <ApiStatus state={apiState} onRetry={retry}/>
      </div>
    </div>
  </header>;
}
