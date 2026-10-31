"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, Check, Database, GitBranch, LockKeyhole, ShieldCheck } from "lucide-react";
import { Header } from "@/components/header";
import { DatasetUpload } from "@/components/dataset-upload";
import { DatasetProfile } from "@/components/dataset-profile";
import { QuestionComposer } from "@/components/question-composer";
import { AnalysisWorkspace } from "@/components/analysis-workspace";
import { readableError, veraApi } from "@/lib/vera-api";
import type { AnalysisResponse, DatasetUploadResponse } from "@/types/vera";

export function AppShell() {
  const [apiState, setApiState] = useState<"checking" | "connected" | "unavailable">("checking");
  const [dataset, setDataset] = useState<DatasetUploadResponse | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const scrollOnResult = useRef(false);

  const checkHealth = useCallback(async () => {
    setApiState("checking");
    try {
      const result = await veraApi.health();
      setApiState(result.status === "ok" ? "connected" : "unavailable");
    } catch { setApiState("unavailable"); }
  }, []);
  useEffect(() => {
    let active = true;
    void veraApi.health().then((result) => {
      if (active) setApiState(result.status === "ok" ? "connected" : "unavailable");
    }).catch(() => {
      if (active) setApiState("unavailable");
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!analysis || !scrollOnResult.current) return;
    scrollOnResult.current = false;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    resultsRef.current?.scrollIntoView({ behavior: reducedMotion ? "instant" : "smooth", block: "start" });
  }, [analysis]);

  async function upload(file: File) {
    setDataset(null); setAnalysis(null); setAnalysisError(null); setUploadError(null);
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setUploadError("Select a CSV (.csv) file. Excel and JSON are not supported."); return;
    }
    if (file.size === 0) { setUploadError("This CSV file is empty."); return; }
    if (file.size > 25 * 1024 * 1024) { setUploadError("The CSV exceeds the 25 MB upload limit."); return; }
    setUploadBusy(true);
    try {
      const result = await veraApi.uploadDataset(file);
      if (!result.dataset_id || !result.profile || !Array.isArray(result.profile.columns)) throw new Error("The API returned an incomplete dataset profile.");
      setDataset(result);
    } catch (error) { setUploadError(readableError(error)); }
    finally { setUploadBusy(false); }
  }

  async function runAnalysis() {
    if (!dataset || !question.trim() || analysisBusy || uploadBusy) return;
    setAnalysisBusy(true); setAnalysis(null); setAnalysisError(null);
    try {
      const result = await veraApi.runAnalysis(dataset.dataset_id, question.trim());
      if (!result.analysis_id || !Array.isArray(result.steps) || !Array.isArray(result.visualizations) || typeof result.succeeded !== "boolean") throw new Error("The API returned an incomplete analysis response.");
      scrollOnResult.current = true;
      setAnalysis(result);
    } catch (error) { setAnalysisError(readableError(error)); }
    finally { setAnalysisBusy(false); }
  }

  return <div className="min-h-screen">
    <Header apiState={apiState} retry={() => void checkHealth()}/>
    <main className="mx-auto max-w-[1320px] px-5 pb-24 pt-11 sm:px-8 sm:pt-14 lg:px-12">
      <div className="mb-11 flex items-center gap-2 text-[11px] font-semibold text-[#819086]"><span className="rounded-md bg-[#e5eee7] px-2.5 py-1 text-[#286b51]">WORKSPACE / 01</span><span className="mx-1 text-[#c2cdc4]">/</span> Investigation studio</div>
      <div className="grid gap-7 border-b border-[#dce4dc] pb-12 lg:grid-cols-[1fr_auto] lg:items-end lg:gap-20">
        <div className="max-w-[800px]"><div className="eyebrow mb-5 flex items-center gap-2"><span className="status-dot status-dot--on"/> PRECISION OVER ASSUMPTION</div>
          <h1 className="text-[clamp(2.7rem,5vw,4.7rem)] font-semibold leading-[1.06] tracking-[-.072em]">Business questions.<br/><span className="text-[#237b5d]">Verified answers.</span></h1>
          <p className="mt-6 max-w-[575px] text-[15px] leading-[1.85] text-[#77857c] sm:text-[16px]">Ask business questions. Get verified analytical evidence. From a raw dataset to a traceable, decision-ready investigation.</p>
        </div>
        <div className="hidden max-w-[260px] border-l border-[#d5dfd5] pl-6 lg:block"><p className="eyebrow mb-3">THE VERA METHOD</p><div className="space-y-2.5 text-[12px] font-semibold text-[#526359]"><p className="flex items-center gap-2"><GitBranch size={14} className="text-[#55947a]"/>AI analytical planning</p><p className="flex items-center gap-2"><Database size={14} className="text-[#55947a]"/>Deterministic analytics</p><p className="flex items-center gap-2"><ShieldCheck size={14} className="text-[#55947a]"/>Evidence verification</p></div></div>
      </div>
      <div className="mb-7 mt-9 flex flex-wrap items-center justify-between gap-3"><div><p className="eyebrow mb-1.5">YOUR INVESTIGATION</p><h2 className="text-[22px] font-bold tracking-[-.055em]">From data to decision.</h2></div><div className="flex items-center gap-2 rounded-full border border-[#dbe5dd] bg-white px-4 py-2 text-[11px] font-semibold text-[#5e7967]"><Check size={13} className="text-[#267f58]"/> Built on verified evidence</div></div>
      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,.92fr)_minmax(0,1.08fr)] lg:gap-6">
        <div className="min-w-0"><DatasetUpload dataset={dataset} busy={uploadBusy || analysisBusy} error={uploadError} onUpload={upload}/>{dataset && <DatasetProfile dataset={dataset}/>}</div>
        <div className="min-w-0"><QuestionComposer question={question} setQuestion={setQuestion} ready={Boolean(dataset) && !uploadBusy} busy={analysisBusy} onRun={() => void runAnalysis()}/></div>
      </div>
      <div ref={resultsRef} className="scroll-mt-8"><AnalysisWorkspace analysis={analysis} busy={analysisBusy} error={analysisError}/></div>
      <div className="mt-14 flex flex-wrap items-center justify-between gap-4 border-t border-[#dce5dc] pt-6"><div className="flex items-center gap-2 text-[11px] text-[#819188]"><LockKeyhole size={14} className="text-[#388366]"/><span>Raw rows stay on the analytics backend. The planner sees safe metadata only.</span></div><div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[.1em] text-[#95a29a]">VERA / VERIFIABLE ANALYTICS <ArrowUpRight size={13}/></div></div>
    </main>
  </div>;
}
