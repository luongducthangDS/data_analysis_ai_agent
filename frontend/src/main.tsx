import React, { useState, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";
import Plot from "react-plotly.js";
import "./styles.css";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Profile {
  rows: number;
  columns: number;
  numeric_summary: Record<string, Record<string, number | null>>;
  missing_values: Record<string, number>;
}
interface ChartSpec {
  chart_id: string;
  title: string;
  chart_type: string;
  plotly_json: { data: unknown[]; layout: Record<string, unknown> };
}
interface Message {
  role: "user" | "assistant";
  content: string;
  charts?: ChartSpec[];
  queries?: string[];
}

// ── API ───────────────────────────────────────────────────────────────────────
const api = {
  async upload(files: File[]) {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const r = await fetch("/api/upload", { method: "POST", body: form });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Upload failed");
    return r.json();
  },
  async analyze(session_id: string, question: string) {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, question }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Analyze failed");
    return r.json();
  },
  async chat(session_id: string, question: string) {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, question }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Chat failed");
    return r.json();
  },
};

// ── Simple markdown renderer ──────────────────────────────────────────────────
function MdText({ text }: { text: string }) {
  return (
    <div className="md">
      {text.split("\n").map((line, i) => {
        if (line.startsWith("### ")) return <h3 key={i}>{line.slice(4)}</h3>;
        if (line.startsWith("## ")) return <h2 key={i}>{line.slice(3)}</h2>;
        if (line.startsWith("# ")) return <h1 key={i}>{line.slice(2)}</h1>;
        if (line.startsWith("⚠️")) return <p key={i} className="warn">{line}</p>;
        if (/^\d+\.\s/.test(line)) return <p key={i} className="li">{line}</p>;
        if (line.trim() === "") return <div key={i} className="br" />;
        return <p key={i}>{line}</p>;
      })}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
function App() {
  const [sessionId, setSessionId] = useState("");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [previewCols, setPreviewCols] = useState<string[]>([]);
  const [previewRows, setPreviewRows] = useState<Record<string, string>[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"preview" | "chat" | "charts">("preview");
  const [allCharts, setAllCharts] = useState<ChartSpec[]>([]);
  const [reportId, setReportId] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function handleFiles(files: FileList | File[]) {
    const arr = Array.from(files).filter((f) =>
      [".csv", ".xlsx", ".xls"].some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (!arr.length) return alert("Only CSV/XLSX/XLS files are supported.");
    setBusy(true);
    try {
      const d = await api.upload(arr);
      setSessionId(d.session_id);
      setProfile(d.profile);
      setPreviewCols(d.preview_columns ?? []);
      setPreviewRows(d.preview_rows ?? []);
      setSuggestions(d.suggested_queries ?? []);
      setMessages([]);
      setAllCharts([]);
      setTab("preview");
    } catch (e: unknown) {
      alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function send(q: string, mode: "analyze" | "chat" = "analyze") {
    if (!q.trim() || !sessionId || busy) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setBusy(true);
    setTab("chat");
    try {
      const d = mode === "analyze"
        ? await api.analyze(sessionId, q)
        : await api.chat(sessionId, q);
      const charts: ChartSpec[] = d.charts ?? [];
      setMessages((m) => [...m, {
        role: "assistant",
        content: d.answer ?? "",
        charts,
        queries: d.executed_queries ?? [],
      }]);
      if (charts.length) { setAllCharts((c) => [...c, ...charts]); setTab("charts"); }
      if (d.report_id) setReportId(d.report_id);
    } catch (e: unknown) {
      setMessages((m) => [...m, { role: "assistant", content: `❌ ${(e as Error).message}` }]);
    } finally {
      setBusy(false);
    }
  }

  const numericCount = profile ? Object.keys(profile.numeric_summary).length : 0;
  const missingCount = profile ? Object.values(profile.missing_values).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
            <rect width="26" height="26" rx="7" fill="#6366f1"/>
            <rect x="5" y="13" width="4" height="8" rx="1" fill="white"/>
            <rect x="11" y="9" width="4" height="12" rx="1" fill="white"/>
            <rect x="17" y="5" width="4" height="16" rx="1" fill="white"/>
          </svg>
          <span>DataAgent</span>
        </div>

        <nav>
          {(["preview", "chat", "charts"] as const).map((t) => (
            <button key={t} className={`nav-item${tab === t ? " active" : ""}`} onClick={() => setTab(t)}>
              {t === "preview" && "📊"}{t === "chat" && "💬"}{t === "charts" && "📈"}&nbsp;
              {t.charAt(0).toUpperCase() + t.slice(1)}
              {t === "charts" && allCharts.length > 0 && <span className="badge">{allCharts.length}</span>}
            </button>
          ))}
        </nav>

        <div className="divider" />

        <div
          className={`drop-zone${dragging ? " dragging" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
          onClick={() => fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" multiple accept=".csv,.xlsx,.xls"
            style={{ display: "none" }} onChange={(e) => e.target.files && handleFiles(e.target.files)} />
          <span className="drop-icon">📁</span>
          <span>{busy && !profile ? "Uploading…" : "Drop CSV / XLSX or click"}</span>
        </div>

        {profile && (
          <div className="info-panel">
            <div className="info-row"><span>Rows</span><b>{profile.rows.toLocaleString()}</b></div>
            <div className="info-row"><span>Columns</span><b>{profile.columns}</b></div>
            <div className="info-row"><span>Numeric</span><b>{numericCount}</b></div>
            <div className="info-row"><span>Missing</span><b>{missingCount}</b></div>
          </div>
        )}

        {suggestions.length > 0 && (
          <div className="suggestions">
            <div className="sugg-label">Suggested queries</div>
            {suggestions.map((s) => (
              <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        )}

        {reportId && (
          <a className="report-link" href={`/api/report/${reportId}`} target="_blank" rel="noreferrer">
            ⬇ Download Report
          </a>
        )}
      </aside>

      {/* Main */}
      <main className="main">
        {/* Preview tab */}
        {tab === "preview" && (
          <div className="panel">
            {!sessionId ? (
              <div className="empty">
                <div className="empty-icon">📂</div>
                <div className="empty-title">No dataset loaded</div>
                <div className="empty-sub">Upload a CSV or Excel file to start</div>
              </div>
            ) : (
              <>
                <div className="panel-head">Data Preview <span className="muted">— first 10 rows</span></div>
                <div className="tbl-wrap">
                  <table>
                    <thead><tr>{previewCols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                    <tbody>
                      {previewRows.map((row, i) => (
                        <tr key={i}>{previewCols.map((c) => <td key={c}>{row[c] ?? ""}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="tbl-foot">
                  {profile?.rows.toLocaleString()} total rows · {profile?.columns} columns
                </div>
              </>
            )}
          </div>
        )}

        {/* Chat tab */}
        {tab === "chat" && (
          <div className="panel chat-panel">
            <div className="messages">
              {messages.length === 0 && (
                <div className="empty">
                  <div className="empty-icon">💬</div>
                  <div className="empty-title">Ask anything about your data</div>
                  <div className="empty-sub">e.g. "tổng amount theo category" · "top 5 employees"</div>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`msg ${msg.role}`}>
                  <div className="bubble">
                    {msg.role === "assistant" ? <MdText text={msg.content} /> : msg.content}
                    {msg.queries && msg.queries.length > 0 && (
                      <details className="plan-detail">
                        <summary>Plan</summary>
                        <pre>{msg.queries.join("\n")}</pre>
                      </details>
                    )}
                  </div>
                </div>
              ))}
              {busy && (
                <div className="msg assistant">
                  <div className="bubble typing"><span/><span/><span/></div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
            <div className="input-bar">
              <textarea rows={2}
                placeholder={sessionId ? "Hỏi về dataset của bạn… (Enter to send)" : "Upload dataset trước"}
                value={question} disabled={!sessionId || busy}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(question); } }}
              />
              <div className="input-btns">
                <button className="btn-primary" disabled={!sessionId || !question.trim() || busy}
                  onClick={() => send(question, "analyze")}>Analyze ↵</button>
                <button className="btn-sec" disabled={!sessionId || !question.trim() || busy}
                  onClick={() => send(question, "chat")}>Quick query</button>
              </div>
            </div>
          </div>
        )}

        {/* Charts tab */}
        {tab === "charts" && (
          <div className="panel">
            {allCharts.length === 0 ? (
              <div className="empty">
                <div className="empty-icon">📈</div>
                <div className="empty-title">No charts yet</div>
                <div className="empty-sub">Ask a question to generate charts</div>
              </div>
            ) : (
              <>
                <div className="panel-head">Charts</div>
                <div className="chart-grid">
                  {allCharts.map((c) => (
                    <div key={c.chart_id} className="chart-card">
                      <div className="chart-title">{c.title}</div>
                      <Plot
                        data={c.plotly_json.data as never}
                        layout={{ ...(c.plotly_json.layout as object), paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" }, margin: { l: 48, r: 16, t: 32, b: 64 } }}
                        useResizeHandler style={{ width: "100%", height: "320px" }}
                        config={{ displayModeBar: false }}
                      />
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
