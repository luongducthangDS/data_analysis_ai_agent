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
interface AgentStep {
  step: number;
  tool_name: string;
  arguments: Record<string, unknown>;
  result_summary: string;
  charts?: ChartSpec[];
}
interface Message {
  role: "user" | "assistant";
  content: string;
  charts?: ChartSpec[];
  queries?: string[];
  agentSteps?: AgentStep[];
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
  async agentChat(session_id: string, question: string) {
    const r = await fetch("/api/agent-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, question }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Agent chat failed");
    return r.json();
  },
  async importUrl(url: string) {
    const r = await fetch("/api/import-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Import failed");
    return r.json();
  },
  async importGSheet(url_or_id: string, sheet_name?: string) {
    const r = await fetch("/api/import-gsheet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url_or_id, sheet_name: sheet_name || undefined }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Google Sheets import failed");
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

// ── Agent Steps display ───────────────────────────────────────────────────────
const TOOL_ICONS: Record<string, string> = {
  analyze_data: "📊",
  query_sql: "🔍",
  get_profile: "🗂️",
  generate_chart: "📈",
};

function AgentStepsPanel({ steps }: { steps: AgentStep[] }) {
  if (!steps.length) return null;
  return (
    <details className="agent-steps">
      <summary>🤖 Agent thực hiện {steps.length} bước</summary>
      <div className="agent-steps-list">
        {steps.map((s) => (
          <div key={s.step} className="agent-step-item">
            <span className="agent-step-icon">{TOOL_ICONS[s.tool_name] ?? "🔧"}</span>
            <div className="agent-step-body">
              <span className="agent-step-name">{s.tool_name}</span>
              <span className="agent-step-summary">{s.result_summary.slice(0, 120)}</span>
            </div>
          </div>
        ))}
      </div>
    </details>
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
  const [importTab, setImportTab] = useState<"file" | "sheets" | "url">("file");
  const [importUrl, setImportUrl] = useState("");
  const [importSheet, setImportSheet] = useState("");
  const [importSheetName, setImportSheetName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  function applyUploadResponse(d: Record<string, unknown>) {
    setSessionId(d.session_id as string);
    setProfile(d.profile as Profile);
    setPreviewCols((d.preview_columns as string[]) ?? []);
    setPreviewRows((d.preview_rows as Record<string, string>[]) ?? []);
    setSuggestions((d.suggested_queries as string[]) ?? []);
    setMessages([]);
    setAllCharts([]);
    setTab("preview");
  }

  async function handleFiles(files: FileList | File[]) {
    const arr = Array.from(files).filter((f) =>
      [".csv", ".xlsx", ".xls"].some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (!arr.length) return alert("Only CSV/XLSX/XLS files are supported.");
    setBusy(true);
    try {
      applyUploadResponse(await api.upload(arr));
    } catch (e: unknown) {
      alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleImportUrl() {
    if (!importUrl.trim()) return;
    setBusy(true);
    try {
      applyUploadResponse(await api.importUrl(importUrl.trim()));
      setImportUrl("");
    } catch (e: unknown) {
      alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleImportGSheet() {
    if (!importSheet.trim()) return;
    setBusy(true);
    try {
      applyUploadResponse(await api.importGSheet(importSheet.trim(), importSheetName.trim() || undefined));
      setImportSheet("");
      setImportSheetName("");
    } catch (e: unknown) {
      alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function send(q: string, mode: "analyze" | "chat" | "agent" = "analyze") {
    if (!q.trim() || !sessionId || busy) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setBusy(true);
    setTab("chat");
    try {
      let d: Record<string, unknown>;
      if (mode === "agent") {
        d = await api.agentChat(sessionId, q);
      } else if (mode === "analyze") {
        d = await api.analyze(sessionId, q);
      } else {
        d = await api.chat(sessionId, q);
      }
      const charts: ChartSpec[] = (d.charts as ChartSpec[]) ?? [];
      const agentSteps: AgentStep[] = (d.agent_steps as AgentStep[]) ?? [];
      setMessages((m) => [...m, {
        role: "assistant",
        content: (d.answer as string) ?? "",
        charts,
        queries: (d.executed_queries as string[]) ?? [],
        agentSteps: agentSteps.length > 0 ? agentSteps : undefined,
      }]);
      if (charts.length) { setAllCharts((c) => [...c, ...charts]); setTab("charts"); }
      if (d.report_id) setReportId(d.report_id as string);
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

        <div className="import-panel">
          <div className="import-tabs">
            {(["file", "sheets", "url"] as const).map((t) => (
              <button key={t} className={`import-tab${importTab === t ? " active" : ""}`}
                onClick={() => setImportTab(t)}>
                {t === "file" && "📁"}{t === "sheets" && "📊"}{t === "url" && "🔗"}
              </button>
            ))}
          </div>

          {importTab === "file" && (
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
          )}

          {importTab === "sheets" && (
            <div className="import-form">
              <input className="import-input" type="text" placeholder="Google Sheets URL hoặc Sheet ID"
                value={importSheet} onChange={(e) => setImportSheet(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleImportGSheet()} />
              <input className="import-input" type="text" placeholder="Tên sheet (để trống = sheet đầu tiên)"
                value={importSheetName} onChange={(e) => setImportSheetName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleImportGSheet()} />
              <button className="import-btn" disabled={!importSheet.trim() || busy}
                onClick={handleImportGSheet}>
                {busy ? "Đang tải…" : "Import từ Sheets"}
              </button>
              <div className="import-hint">Sheet phải được chia sẻ với email service account</div>
            </div>
          )}

          {importTab === "url" && (
            <div className="import-form">
              <input className="import-input" type="text"
                placeholder="Dán URL file CSV, XLSX hoặc Google Sheets…"
                value={importUrl} onChange={(e) => setImportUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleImportUrl()} />
              <button className="import-btn" disabled={!importUrl.trim() || busy}
                onClick={handleImportUrl}>
                {busy ? "Đang tải…" : "Import từ URL"}
              </button>
              <div className="import-hint">Hỗ trợ CSV public, Google Sheets, Dropbox shared links</div>
            </div>
          )}
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
                    {msg.agentSteps && <AgentStepsPanel steps={msg.agentSteps} />}
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
                <button className="btn-agent" disabled={!sessionId || !question.trim() || busy}
                  title="ReAct Agent — tự chọn và gọi nhiều tools, hiển thị từng bước"
                  onClick={() => send(question, "agent")}>🤖 Agent</button>
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
