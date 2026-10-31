"use client";

import { ArrowRight, Command, FileQuestion, LoaderCircle } from "lucide-react";

export const OFFICIAL_QUESTION = "Why did our sales decline, and did the September campaign actually help?";

export function QuestionComposer({ question, setQuestion, ready, busy, onRun }: {
  question: string; setQuestion: (value: string) => void;
  ready: boolean; busy: boolean; onRun: () => void;
}) {
  const canRun = ready && !busy && Boolean(question.trim());
  return <section aria-labelledby="question-heading" className="surface overflow-hidden">
    <div className="flex items-start justify-between border-b border-[#ebefeb] px-6 py-6 sm:px-7">
      <div><div className="eyebrow mb-2">02 / INVESTIGATE</div><h2 id="question-heading" className="text-xl font-bold tracking-[-.04em]">Ask VERA</h2><p className="mt-1 text-[13px] text-[#829088]">Turn a question into verifiable evidence.</p></div>
      <div className="flex size-10 items-center justify-center rounded-xl bg-[#edf3ef] text-[#296d56]"><FileQuestion size={19}/></div>
    </div>
    <div className="px-5 py-6 sm:px-7">
      <label htmlFor="business-question" className="mb-3 block text-[13px] font-bold">What do you want to understand?</label>
      <div className="overflow-hidden rounded-xl border border-[#d8e0d8] bg-[#fcfdfb] focus-within:border-[#3a9777] focus-within:ring-2 focus-within:ring-[#d4eee2]">
        <textarea id="business-question" rows={5} placeholder={ready ? "What changed in our business, and what does the evidence show?" : "Upload a CSV to begin your investigation…"} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => {if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {event.preventDefault(); if (canRun) onRun();}}} disabled={!ready || busy} className="w-full resize-y bg-transparent px-4 py-4 text-[14px] leading-[1.75] text-[#1e3128] outline-none placeholder:text-[#a0ada4] disabled:cursor-not-allowed disabled:opacity-65"/>
        <div className="flex items-center justify-end gap-1 border-t border-[#eef1ec] px-4 py-2 text-[10px] text-[#93a097]"><Command size={12}/> / Ctrl + Enter to run</div>
      </div>
      <div className="mt-5"><p className="eyebrow mb-3">Try the official question</p><button type="button" disabled={!ready || busy} onClick={() => setQuestion(OFFICIAL_QUESTION)} className="group flex w-full items-start justify-between gap-3 rounded-xl border border-[#e2e8e1] bg-[#f7faf7] px-4 py-3.5 text-left transition hover:border-[#aac9b9] hover:bg-[#f0f7f1] disabled:opacity-50"><span className="text-[12px] font-medium leading-5 text-[#476251]">{OFFICIAL_QUESTION}</span><ArrowRight size={16} className="mt-0.5 shrink-0 text-[#398366] transition group-hover:translate-x-0.5"/></button></div>
      <button type="button" disabled={!canRun} onClick={onRun} className="mt-6 flex min-h-14 w-full items-center justify-center gap-3 rounded-xl bg-[#176b57] px-5 text-[13px] font-bold text-white shadow-[0_7px_17px_rgba(23,107,87,.15)] transition hover:bg-[#10533f] disabled:bg-[#b4c9bc] disabled:shadow-none">
        {busy ? <><LoaderCircle className="animate-spin" size={18}/>Running verified investigation…</> : <>Run verified investigation <ArrowRight size={17}/></>}
      </button>
      <p className="mt-3 text-center text-[11px] text-[#96a29b]">{busy ? "Planning and deterministic checks are running. The API does not stream progress." : "VERA plans, executes and verifies the analysis on the backend."}</p>
    </div>
  </section>;
}
