import React, { useState, useRef, useEffect, useCallback } from "react";
import { createRoot } from "react-dom/client";
import Plot from "react-plotly.js";
import "./styles.css";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Profile {
  rows: number;
  columns: number;
  column_types: Record<string, string>;
  numeric_summary: Record<string, Record<string, number | null>>;
  categorical_summary: Record<string, Array<{ value: string; count: number }>>;
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
  source?: "llm" | "fallback" | "bot_info" | "off_topic";
  streaming?: boolean;
  nodes?: string[];   // graph nodes completed so far (while streaming)
}

interface KPICard {
  label: string;
  value: string;
  delta: string | null;
  delta_positive: boolean | null;
  formula: string;
  is_alert: boolean;
}
interface DashboardData {
  session_id: string;
  platform: string | null;
  kpi_cards: KPICard[];
  charts: ChartSpec[];
  top_products: Record<string, unknown>[];
  col_map: Record<string, string>;
  unmapped_cols: string[];
  is_ecommerce: boolean;
  suggested_queries: string[];
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
        if (line.startsWith("- ") || line.startsWith("* ")) return <p key={i} className="li">{line.slice(2)}</p>;
        if (line.trim() === "") return <div key={i} className="br" />;
        return <p key={i}>{line}</p>;
      })}
    </div>
  );
}

// ── Source badge ──────────────────────────────────────────────────────────────
const SOURCE_LABELS: Record<string, { label: string; title: string; cls: string }> = {
  llm:       { label: "AI Synthesis",      title: "Answer synthesized by LLM from analysis results", cls: "badge-llm"      },
  fallback:  { label: "Deterministic",     title: "LLM unavailable — answer built from raw data directly", cls: "badge-fallback" },
  bot_info:  { label: "Bot Info",          title: "Agent self-description, not data analysis", cls: "badge-bot"      },
  off_topic: { label: "Off Topic",         title: "Question outside analysis scope", cls: "badge-bot"      },
};

function SourceBadge({ source }: { source: string }) {
  const info = SOURCE_LABELS[source] ?? { label: source, title: source, cls: "badge-bot" };
  return <span className={`source-badge ${info.cls}`} title={info.title}>{info.label}</span>;
}

// ── Plan detail — human-readable execution plan ───────────────────────────────
function PlanDetail({ queries }: { queries: string[] }) {
  if (!queries || queries.length === 0) return null;

  const renderPlan = (raw: string) => {
    try {
      const p = JSON.parse(raw);
      const chips: string[] = [];
      if (p.action)        chips.push(`action: ${p.action}`);
      if (p.metric_col)    chips.push(`metric: ${p.metric_col}`);
      if (p.group_by?.length)  chips.push(`group by: ${p.group_by.join(", ")}`);
      if (p.agg)           chips.push(`agg: ${p.agg}`);
      if (p.sort)          chips.push(`sort: ${p.sort}`);
      if (p.limit)         chips.push(`top: ${p.limit}`);
      if (p.filters?.length)   chips.push(`filter: ${p.filters.map((f: Record<string, unknown>) => `${f.column}=${f.value}`).join(", ")}`);
      if (p.derived_columns?.length) chips.push(`derived: ${p.derived_columns.map((d: Record<string, unknown>) => d.name).join(", ")}`);
      return chips.length ? chips : [raw];
    } catch {
      return [raw];
    }
  };

  return (
    <details className="plan-detail">
      <summary>Kế hoạch phân tích</summary>
      <div className="plan-chips">
        {queries.map((q, i) => (
          <div key={i} className="plan-step">
            {renderPlan(q).map((chip, j) => (
              <span key={j} className="plan-chip">{chip}</span>
            ))}
          </div>
        ))}
      </div>
    </details>
  );
}

// ── Node progress strip ───────────────────────────────────────────────────────
const NODE_LABELS: Record<string, string> = {
  classify:   "Phân loại",
  planner:    "Lập kế hoạch",
  execute:    "Thực thi",
  synthesize: "Tổng hợp",
  bot_info:   "Bot Info",
  off_topic:  "Off Topic",
};

function NodeProgress({ nodes }: { nodes: string[] }) {
  if (!nodes.length) return null;
  return (
    <div className="node-progress">
      {nodes.map((n) => (
        <span key={n} className="node-chip done">{NODE_LABELS[n] ?? n} ✓</span>
      ))}
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

// ── KPI Card ─────────────────────────────────────────────────────────────────
function KPICardComponent({ card }: { card: KPICard }) {
  return (
    <div className={`kpi-card${card.is_alert ? " kpi-alert" : ""}`}>
      <div className="kpi-label">{card.label}</div>
      <div className="kpi-value">{card.value}</div>
      {card.delta && (
        <div className={`kpi-delta ${card.delta_positive ? "positive" : "negative"}`}>
          {card.delta_positive ? "▲" : "▼"} {card.delta}
        </div>
      )}
      {card.formula && (
        <details className="kpi-formula">
          <summary>Xem công thức</summary>
          <span>{card.formula}</span>
        </details>
      )}
    </div>
  );
}

// ── Dashboard Panel ───────────────────────────────────────────────────────────
function DashboardPanel({
  data,
  sessionId,
  onAsk,
}: {
  data: DashboardData;
  sessionId: string;
  onAsk: (q: string) => void;
}) {
  const [subTab, setSubTab] = React.useState<"kpi" | "products" | "trends">("kpi");
  const trendChart = data.charts.find((c) => c.chart_type === "line");
  const otherCharts = data.charts.filter((c) => c.chart_type !== "line");

  return (
    <div className="panel dashboard-panel">
      <div className="panel-head">
        Dashboard
        {data.platform && (
          <span className="platform-badge">{data.platform.charAt(0).toUpperCase() + data.platform.slice(1)}</span>
        )}
        <a className="export-xlsx-btn" href={`/api/dashboard/${sessionId}/export.xlsx`} download>
          ⬇ Xuất Excel
        </a>
      </div>

      {data.unmapped_cols.length > 0 && (
        <div className="unmapped-notice">
          ⚠ Không nhận diện được: {data.unmapped_cols.join(", ")} — hỏi chatbot để phân tích thủ công.
        </div>
      )}

      {/* Sub-tabs */}
      <div className="dashboard-subtabs">
        {(["kpi", "products", "trends"] as const).map((t) => (
          <button
            key={t}
            className={`dash-tab${subTab === t ? " active" : ""}`}
            onClick={() => setSubTab(t)}
          >
            {t === "kpi" && "📊 KPI"}
            {t === "products" && "🏆 Top 10"}
            {t === "trends" && "📈 Xu hướng"}
          </button>
        ))}
      </div>

      {/* KPI sub-tab */}
      {subTab === "kpi" && (
        <>
          <div className="kpi-grid">
            {data.kpi_cards.map((card) => (
              <KPICardComponent key={card.label} card={card} />
            ))}
          </div>
          {/* Non-trend charts (bar charts: top products, by platform) */}
          {otherCharts.length > 0 && (
            <div className="chart-grid" style={{ marginTop: 24 }}>
              {otherCharts.map((c) => (
                <div key={c.chart_id} className="chart-card">
                  <div className="chart-title">{c.title}</div>
                  <Plot
                    data={c.plotly_json.data as never}
                    layout={{ ...(c.plotly_json.layout as object), paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" }, margin: { l: 120, r: 16, t: 32, b: 48 } }}
                    useResizeHandler style={{ width: "100%", height: "320px" }}
                    config={{ displayModeBar: false }}
                  />
                  <a
                    className="chart-export-link"
                    href={`/api/dashboard/${sessionId}/export-chart/${c.chart_id}.png`}
                    download
                  >
                    ⬇ Tải ảnh
                  </a>
                </div>
              ))}
            </div>
          )}
          {/* AI-generated suggested queries */}
          {data.suggested_queries?.length > 0 && (
            <div className="suggestions" style={{ marginTop: 20 }}>
              <div className="sugg-label">Phân tích sâu hơn</div>
              {data.suggested_queries.map((q) => (
                <button key={q} className="chip" onClick={() => onAsk(q)}>{q}</button>
              ))}
            </div>
          )}
        </>
      )}

      {/* Products sub-tab */}
      {subTab === "products" && (
        <div className="tbl-wrap">
          {data.top_products.length === 0 ? (
            <div className="empty">
              <div className="empty-sub">Không có dữ liệu phân nhóm (cần cột phân loại)</div>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Nhóm</th>
                  <th>Giá trị</th>
                </tr>
              </thead>
              <tbody>
                {data.top_products.map((p, i) => (
                  <tr key={i}>
                    <td>{String(p.rank ?? i + 1)}</td>
                    <td>{String(p.name ?? "")}</td>
                    <td>{typeof p.value === "number" ? p.value.toLocaleString("vi-VN") : String(p.value ?? "—")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Trends sub-tab */}
      {subTab === "trends" && (
        <div>
          {trendChart ? (
            <div className="chart-card" style={{ marginTop: 8 }}>
              <div className="chart-title">{trendChart.title}</div>
              <Plot
                data={trendChart.plotly_json.data as never}
                layout={{ ...(trendChart.plotly_json.layout as object), paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" }, margin: { l: 56, r: 16, t: 32, b: 64 } }}
                useResizeHandler style={{ width: "100%", height: "360px" }}
                config={{ displayModeBar: false }}
              />
              <a
                className="chart-export-link"
                href={`/api/dashboard/${sessionId}/export-chart/${trendChart.chart_id}.png`}
                download
              >
                ⬇ Tải ảnh
              </a>
            </div>
          ) : (
            <div className="empty">
              <div className="empty-sub">Không có dữ liệu xu hướng (cần cột ngày đặt hàng)</div>
            </div>
          )}
        </div>
      )}
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
  const [tab, setTab] = useState<"dashboard" | "preview" | "chat" | "charts">("preview");
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
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
    const sid = d.session_id as string;
    setSessionId(sid);
    setProfile(d.profile as Profile);
    setPreviewCols((d.preview_columns as string[]) ?? []);
    setPreviewRows((d.preview_rows as Record<string, string>[]) ?? []);
    setSuggestions((d.suggested_queries as string[]) ?? []);
    setMessages([]);
    setAllCharts([]);
    setDashboardData(null);
    setTab("preview");

    // Auto-fetch AI dashboard — always switch to dashboard tab when ready
    fetch(`/api/dashboard/${sid}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((dash: DashboardData | null) => {
        if (!dash || !dash.kpi_cards?.length) return;
        setDashboardData(dash);
        setTab("dashboard");
      })
      .catch(() => {});
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

  // Streaming send — all modes use /api/chat/stream
  const send = useCallback(async (q: string) => {
    if (!q.trim() || !sessionId || busy) return;

    // Append user message
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setBusy(true);
    setTab("chat");

    // Append placeholder streaming assistant message
    const assistantIdx = await new Promise<number>((resolve) => {
      setMessages((m) => {
        resolve(m.length);  // will be appended at this index
        return [...m, { role: "assistant", content: "", streaming: true, nodes: [] }];
      });
    });

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question: q }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Stream failed" }));
        throw new Error(err.detail ?? "Stream failed");
      }

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;
          let event: Record<string, unknown>;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          if (event.type === "node") {
            setMessages((m) => m.map((msg, i) =>
              i === assistantIdx
                ? { ...msg, nodes: [...(msg.nodes ?? []), event.node as string] }
                : msg
            ));
          } else if (event.type === "token") {
            setMessages((m) => m.map((msg, i) =>
              i === assistantIdx
                ? { ...msg, content: msg.content + (event.content as string) }
                : msg
            ));
          } else if (event.type === "done") {
            const charts = (event.charts as ChartSpec[]) ?? [];
            const queries = (event.executed_queries as string[]) ?? [];
            const source = (event.source as Message["source"]) ?? "llm";
            setMessages((m) => m.map((msg, i) =>
              i === assistantIdx
                ? { ...msg, streaming: false, charts, queries, source }
                : msg
            ));
            if (charts.length) setAllCharts((c) => [...c, ...charts]);
          } else if (event.type === "error") {
            setMessages((m) => m.map((msg, i) =>
              i === assistantIdx
                ? { ...msg, streaming: false, content: `❌ ${event.detail}` }
                : msg
            ));
          }
        }
      }
    } catch (e: unknown) {
      setMessages((m) => m.map((msg, i) =>
        i === assistantIdx
          ? { ...msg, streaming: false, content: `❌ ${(e as Error).message}` }
          : msg
      ));
    } finally {
      setBusy(false);
    }
  }, [sessionId, busy]);

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
          {dashboardData && dashboardData.kpi_cards?.length > 0 && (
            <button className={`nav-item${tab === "dashboard" ? " active" : ""}`} onClick={() => setTab("dashboard")}>
              📊&nbsp;Dashboard
            </button>
          )}
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

        {sessionId && (
          <a className="report-link" href={`/api/session/${sessionId}/data.csv`} download>
            ⬇ Export CSV
          </a>
        )}
        {reportId && (
          <a className="report-link" href={`/api/report/${reportId}`} target="_blank" rel="noreferrer">
            ⬇ Download Report
          </a>
        )}
      </aside>

      {/* Main */}
      <main className="main">
        {/* Dashboard tab */}
        {tab === "dashboard" && dashboardData && (
          <DashboardPanel
            data={dashboardData}
            sessionId={sessionId}
            onAsk={(q) => { send(q); setTab("chat"); }}
          />
        )}

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
                {profile?.column_types && (
                  <details className="col-types-detail">
                    <summary>Column types</summary>
                    <div className="col-types-grid">
                      {Object.entries(profile.column_types).map(([col, dtype]) => {
                        const missing = profile.missing_values?.[col] ?? 0;
                        const typeClass = dtype.startsWith("int") || dtype.startsWith("float") ? "dtype-num"
                          : dtype.startsWith("datetime") ? "dtype-date"
                          : "dtype-cat";
                        return (
                          <div key={col} className="col-type-row">
                            <span className="col-name">{col}</span>
                            <span className={`col-dtype ${typeClass}`}>{dtype}</span>
                            {missing > 0 && <span className="col-missing">{missing} missing</span>}
                          </div>
                        );
                      })}
                    </div>
                  </details>
                )}
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
                  {msg.role === "user" ? (
                    <div className="bubble">{msg.content}</div>
                  ) : (
                    <div className="assistant-msg">
                      <div className="bubble">
                        {/* Node progress (while streaming) */}
                        {msg.nodes && msg.nodes.length > 0 && (
                          <NodeProgress nodes={msg.nodes} />
                        )}
                        {/* Answer text */}
                        {msg.content ? (
                          <MdText text={msg.content} />
                        ) : msg.streaming ? (
                          <div className="typing"><span/><span/><span/></div>
                        ) : null}
                        {/* Streaming cursor */}
                        {msg.streaming && msg.content && (
                          <span className="cursor">▋</span>
                        )}
                        {/* Source badge — shown after streaming done */}
                        {!msg.streaming && msg.source && (
                          <div className="msg-meta">
                            <SourceBadge source={msg.source} />
                          </div>
                        )}
                        {msg.agentSteps && <AgentStepsPanel steps={msg.agentSteps} />}
                        {msg.queries && msg.queries.length > 0 && (
                          <PlanDetail queries={msg.queries} />
                        )}
                      </div>
                      {msg.charts && msg.charts.length > 0 && (
                        <div className="msg-charts">
                          {msg.charts.map((c) => (
                            <div key={c.chart_id} className="chart-card">
                              <div className="chart-title">{c.title}</div>
                              <Plot
                                data={c.plotly_json.data as never}
                                layout={{ ...(c.plotly_json.layout as object), paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" }, margin: { l: 48, r: 16, t: 32, b: 48 } }}
                                useResizeHandler style={{ width: "100%", height: "280px" }}
                                config={{ displayModeBar: false }}
                              />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
            <div className="input-bar">
              <textarea rows={2}
                placeholder={sessionId ? "Hỏi về dataset… Enter gửi, Shift+Enter xuống dòng" : "Upload dataset trước để bắt đầu"}
                value={question} disabled={!sessionId || busy}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(question); } }}
              />
              <div className="input-btns">
                <button className="btn-primary" disabled={!sessionId || !question.trim() || busy}
                  onClick={() => send(question)}>
                  {busy ? "⏳ Đang xử lý…" : "📊 Phân tích"}
                </button>
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
