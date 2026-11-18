'use client';

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import Sidebar from '@/components/Sidebar';
import TopBar from '@/components/TopBar';
import { fetchJson, type AnalysisMeta } from '@/lib/api';

interface Ctx {
  analyses: AnalysisMeta[];
  analysisId: string;
  setAnalysisId: (id: string) => void;
  live: boolean;
  reload: () => void;
}

const AnalysisCtx = createContext<Ctx>({
  analyses: [], analysisId: '', setAnalysisId: () => {}, live: false, reload: () => {},
});

export const useAnalyses = () => useContext(AnalysisCtx);

export default function Shell({ children }: { children: ReactNode }) {
  const [analyses, setAnalyses] = useState<AnalysisMeta[]>([]);
  const [analysisId, setAnalysisId] = useState('');
  const [live, setLive] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { data, live } = await fetchJson<AnalysisMeta[]>('/api/analyses', 'analyses.json');
        if (cancelled) return;
        setAnalyses(data);
        setLive(live);
        const stored = typeof window !== 'undefined' ? localStorage.getItem('fs.analysis') : null;
        if (stored && data.some((a) => a.id === stored)) {
          setAnalysisId((prev) => prev || stored);
        }
      } catch {
        /* no backend or empty */
      }
    })();
    return () => { cancelled = true; };
  }, [tick]);

  // poll while any analysis is processing
  useEffect(() => {
    const processing = analyses.some((a) => a.status === 'processing' || a.status === 'queued');
    if (!processing) return;
    const t = setInterval(() => setTick((x) => x + 1), 3000);
    return () => clearInterval(t);
  }, [analyses]);

  return (
    <AnalysisCtx.Provider
      value={{
        analyses,
        analysisId,
        setAnalysisId: (id) => {
          setAnalysisId(id);
          if (typeof window !== 'undefined') {
            if (id) {
              localStorage.setItem('fs.analysis', id);
            } else {
              localStorage.removeItem('fs.analysis');
            }
          }
        },
        live,
        reload: () => setTick((x) => x + 1),
      }}
    >
      <div className="shell">
        <Sidebar live={live} />
        <div className="main">
          <TopBar />
          <div className="content">{children}</div>
        </div>
      </div>
    </AnalysisCtx.Provider>
  );
}
