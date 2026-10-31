"use client";

import { useRef, useState } from "react";
import { ArrowDownToLine, ArrowUpFromLine, CheckCircle2, FileText, LoaderCircle, Upload, Play } from "lucide-react";
import type { DatasetUploadResponse } from "@/types/vera";

export function DatasetUpload({ dataset, busy, error, onUpload, onUseDemo, demoBusy }: {
  dataset: DatasetUploadResponse | null; busy: boolean; error: string | null; onUpload: (file: File) => void;
  onUseDemo: () => void; demoBusy: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [pickedFile, setPickedFile] = useState<{ name: string; size: number } | null>(null);

  function selectFile(file?: File) {
    if (!file) return;
    setPickedFile({ name: file.name, size: file.size });
    onUpload(file);
  }

  return <section aria-labelledby="dataset-heading" className="surface overflow-hidden">
    <div className="flex items-start justify-between gap-4 border-b border-[#ebefeb] px-6 py-6 sm:px-7">
      <div>
        <div className="eyebrow mb-2">01 / SOURCE</div>
        <h2 id="dataset-heading" className="text-xl font-bold tracking-[-.04em]">Your dataset</h2>
        <p className="mt-1 text-[13px] text-[#829088]">Start with the evidence.</p>
      </div>
      <div className="flex size-10 items-center justify-center rounded-xl bg-[#edf3ef] text-[#296d56]"><Upload size={19}/></div>
    </div>
    <div className="px-5 py-5 sm:px-7 sm:py-6">
      <input ref={inputRef} id="dataset-csv-file" type="file" accept=".csv,text/csv" className="sr-only" onChange={(event) => {selectFile(event.currentTarget.files?.[0]); event.currentTarget.value = "";}} disabled={busy}/>
      <div role="group" aria-label="CSV dataset upload" onDragEnter={(event) => {event.preventDefault(); if (!busy) setDragging(true);}} onDragOver={(event) => {event.preventDefault(); if (!busy) setDragging(true);}} onDragLeave={(event) => {event.preventDefault(); if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);}} onDrop={(event) => {event.preventDefault(); setDragging(false); if (!busy) selectFile(event.dataTransfer.files?.[0]);}}
        className={`input-card rounded-[13px] border border-dashed px-5 py-7 text-center sm:py-8 ${dragging ? "is-dragging" : "border-[#cedad0] bg-[#fafcf9]"}`}>
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-[13px] border border-[#d7e4d9] bg-white text-[#256b56] shadow-sm">{busy ? <LoaderCircle size={22} className="animate-spin"/> : dataset ? <CheckCircle2 size={23}/> : <ArrowUpFromLine size={22}/>}</div>
        <p className="text-sm font-semibold text-[#263a30]">{busy ? "Uploading your dataset…" : dataset ? "Dataset uploaded" : "Drop a CSV file here"}</p>
        <p className="mt-1.5 text-xs text-[#8b988f]">{busy ? "Validating and profiling your data" : "or choose one from your device"}</p>
        <button type="button" onClick={() => inputRef.current?.click()} disabled={busy} className="mt-5 inline-flex h-10 items-center gap-2 rounded-lg border border-[#cbd9ce] bg-white px-4 text-[12px] font-bold text-[#265c49] shadow-sm transition hover:border-[#176b57] hover:bg-[#f0f6f0] disabled:opacity-50">
          <Upload size={14}/>{dataset ? "Replace CSV" : "Browse files"}
        </button>
        <p className="mt-4 text-[10px] font-medium uppercase tracking-[.11em] text-[#9ba69f]">CSV only · Max 25 MB</p>
      </div>
      {pickedFile && <div className="mt-3 flex min-w-0 items-center gap-3 rounded-xl border border-[#e4ebe4] bg-white p-3">
        <div className="rounded-lg bg-[#eff4f0] p-2 text-[#397560]"><FileText size={16}/></div>
        <div className="min-w-0 flex-1"><p className="truncate text-[12px] font-semibold">{pickedFile.name}</p><p className="text-[11px] text-[#8a978e]">{(pickedFile.size / 1024).toLocaleString(undefined, {maximumFractionDigits: 1})} KB</p></div>
        {dataset && !busy && <CheckCircle2 size={16} className="text-[#23855f]"/>}
      </div>}
      {error && <p className="form-error mt-4" role="alert">{error}</p>}
      <div className="mt-5 border-t border-[#e7ece7] pt-5">
        <button type="button" disabled={busy} onClick={onUseDemo} className="flex w-full items-center justify-center gap-2.5 rounded-xl bg-[#176b57] px-4 py-3.5 text-[13px] font-bold text-white transition hover:bg-[#10533f] disabled:opacity-60">{demoBusy ? <LoaderCircle size={17} className="animate-spin"/> : <Play size={16}/>} {demoBusy ? "Loading official demo…" : "Use demo dataset"}</button>
        <a href="/demo/vera_sales_campaign_demo.csv" download="vera_sales_campaign_demo.csv" className="group mt-3 inline-flex items-center gap-2 text-[11px] font-semibold text-[#607b68] hover:text-[#164936]"><ArrowDownToLine size={14}/>Download CSV <span aria-hidden="true">↗</span></a>
        <p className="mt-2 text-[11px] leading-5 text-[#8c978f]">Synthetic demonstration data · Uploaded through the same API as any CSV</p>
      </div>
    </div>
  </section>;
}
