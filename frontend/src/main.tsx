import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import Plot from "react-plotly.js";
import { BarChart3, FileUp, MessageSquareText } from "lucide-react";
import "./styles.css";

type Profile = {
  rows: number;
  columns: number;
  column_types: Record<string, string>;
  missing_values: Record<string, number>;
  numeric_summary: Record<string, Record<string, number | null>>;
  categorical_summary: Record<string, Array<{ value: string; count: number }>>;
};

type ChartSpec = {
  chart_id: string;
  title: string;
  chart_type: string;
  plotly_json: { data: unknown[]; layout: Record<string, unknown> };
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("Upload a CSV/XLSX file to start.");
  const [charts, setCharts] = useState<ChartSpec[]>([]);
  const [reportId, setReportId] = useState("");
  const [busy, setBusy] = useState(false);

  async function upload() {
    if (!file) return;
    setBusy(true);
    setAnswer("Uploading and profiling dataset...");
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    setBusy(false);
    if (!res.ok) {
      setAnswer(data.detail ?? "Upload failed.");
      return;
    }
    setSessionId(data.session_id);
    setProfile(data.profile);
    setAnswer(JSON.stringify(data.profile, null, 2));
  }

  async function analyze() {
    setBusy(true);
    setAnswer("Generating charts and insights...");
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question: question || null }),
    });
    const data = await res.json();
    setBusy(false);
    if (!res.ok) {
      setAnswer(data.detail ?? "Analyze failed.");
      return;
    }
    setAnswer(data.answer);
    setCharts(data.charts ?? []);
    setReportId(data.report_id);
  }

  async function chat() {
    setBusy(true);
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });
    const data = await res.json();
    setBusy(false);
    if (!res.ok) {
      setAnswer(data.detail ?? "Query failed.");
      return;
    }
    setAnswer(`${data.answer}\n\nSQL:\n${(data.executed_queries ?? []).join("\n")}`);
  }

  return (
    <main>
      <aside>
        <div className="brand">
          <BarChart3 size={24} />
          <div>
            <h1>Data Analysis AI Agent</h1>
            <p>CSV/XLSX insight workflow</p>
          </div>
        </div>

        <section>
          <h2><FileUp size={18} /> Dataset</h2>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <button disabled={!file || busy} onClick={upload}>Upload & Profile</button>
          {profile && (
            <div className="stats">
              <span>{profile.rows.toLocaleString()} rows</span>
              <span>{profile.columns.toLocaleString()} columns</span>
            </div>
          )}
        </section>

        <section>
          <h2><MessageSquareText size={18} /> Ask</h2>
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Tổng sales, top category, trung bình age..." />
          <button disabled={!sessionId || busy} onClick={analyze}>Auto Analyze</button>
          <button disabled={!sessionId || !question || busy} onClick={chat}>Ask Dataset</button>
          {reportId && <a href={`/api/report/${reportId}`} target="_blank">Download Markdown report</a>}
        </section>
      </aside>

      <section className="workspace">
        <h2>Insight Output</h2>
        <pre>{answer}</pre>
        <div className="chartGrid">
          {charts.map((chart) => (
            <div className="chartCard" key={chart.chart_id}>
              <Plot data={chart.plotly_json.data as never} layout={chart.plotly_json.layout} useResizeHandler style={{ width: "100%", height: "360px" }} />
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);

