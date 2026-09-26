import React, { useState, useRef, useEffect, useCallback, useMemo, lazy, memo, Suspense } from "react";
import { createRoot } from "react-dom/client";
import {
  Check, CircleAlert, Download, Eye, EyeOff, FileSpreadsheet, FileText, GitMerge, KeyRound,
  LayoutDashboard, LineChart, Menu, MessageSquare, Plus, Send, ShieldCheck, Square, Store, Table2, TriangleAlert, Upload, X,
} from "lucide-react";
import "./styles.css";

// Plotly is the heaviest chunk — load it only when the first chart renders, not on the welcome screen.
const Plot = lazy(() => import("./Plot"));

// Error bodies aren't always JSON (Render returns an HTML 502 page) — never let r.json() throw.
async function errorDetail(r: Response, fallback: string): Promise<string> {
  const body = await r.json().catch(() => null);
  return (body && typeof body.detail === "string" && body.detail) || `${fallback} (mã ${r.status})`;
}

// ── LLM Keys (localStorage) ───────────────────────────────────────────────────
const LS = {
  get: (k: string) => localStorage.getItem(k) ?? "",
  set: (k: string, v: string) => v ? localStorage.setItem(k, v) : localStorage.removeItem(k),
};

function loadKeys() {
  return {
    gemini:    LS.get("da_gemini_key"),
    anthropic: LS.get("da_anthropic_key"),
    provider:  LS.get("da_provider") || "auto",
  };
}

function saveKeys(keys: ReturnType<typeof loadKeys>) {
  LS.set("da_gemini_key",    keys.gemini);
  LS.set("da_anthropic_key", keys.anthropic);
  LS.set("da_provider",      keys.provider === "auto" ? "" : keys.provider);
}

function keyHeaders(keys: ReturnType<typeof loadKeys>): Record<string, string> {
  const h: Record<string, string> = {};
  if (keys.gemini)    h["X-GEMINI-Key"]     = keys.gemini;
  if (keys.anthropic) h["X-ANTHROPIC-Key"]  = keys.anthropic;
  if (keys.provider && keys.provider !== "auto") h["X-LLM-Provider"] = keys.provider;
  return h;
}

// ── Settings modal ────────────────────────────────────────────────────────────
function SettingsModal({
  onClose,
}: {
  onClose: () => void;
}) {
  const [keys, setKeys] = useState(loadKeys);
  const [reveal, setReveal] = useState({ gemini: false, anthropic: false });
  const [saved, setSaved] = useState(false);
  const ref = useRef<HTMLDialogElement>(null);

  // Native modal <dialog>: Esc closes, background is inert, focus stays inside.
  useEffect(() => { ref.current?.showModal(); }, []);

  function handleSave() {
    saveKeys(keys);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <dialog ref={ref} className="modal" aria-labelledby="settings-title" onClose={onClose}
      onClick={(e) => e.target === e.currentTarget && ref.current?.close()}>
      <div className="modal-inner">
        <div className="modal-head">
          <div>
            <h2 className="modal-title" id="settings-title">Khóa API</h2>
            <p className="modal-sub">Dùng khóa của riêng bạn thay cho khóa máy chủ.</p>
          </div>
          <button className="modal-close" onClick={() => ref.current?.close()} aria-label="Đóng"><X size={16} /></button>
        </div>

        <fieldset className="provider-group">
          <legend>Nhà cung cấp LLM</legend>
          <div className="provider-options">
            {[
              { value: "auto", label: "Tự động" },
              { value: "gemini", label: "Gemini" },
              { value: "anthropic", label: "Anthropic" },
            ].map((o) => (
              <label key={o.value} className={`provider-option${keys.provider === o.value ? " checked" : ""}`}>
                <input
                  type="radio"
                  name="provider"
                  value={o.value}
                  checked={keys.provider === o.value}
                  onChange={() => setKeys((k) => ({ ...k, provider: o.value }))}
                />
                {o.label}
              </label>
            ))}
          </div>
          <div className="settings-hint">Tự động thử Gemini trước, rồi đến Anthropic.</div>
        </fieldset>

        <div className="settings-group">
          {[
            { id: "gemini",    label: "Gemini API key",    tag: "Free tier", placeholder: "AIza…",    recommended: true },
            { id: "anthropic", label: "Anthropic API key", tag: "Trả phí",   placeholder: "sk-ant-…", recommended: false },
          ].map(({ id, label, tag, placeholder, recommended }) => (
            <div className="settings-field" key={id}>
              <label htmlFor={`key-${id}`}>
                {label}
                <span className={`tag${recommended ? " recommended" : ""}`}>{tag}</span>
              </label>
              <div className="settings-input-wrap">
                <input
                  id={`key-${id}`}
                  className="settings-input"
                  type={reveal[id as keyof typeof reveal] ? "text" : "password"}
                  placeholder={placeholder}
                  value={keys[id as keyof typeof keys]}
                  onChange={(e) => setKeys((k) => ({ ...k, [id]: e.target.value }))}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  className="toggle-reveal"
                  type="button"
                  onClick={() => setReveal((r) => ({ ...r, [id]: !r[id as keyof typeof r] }))}
                  aria-label={reveal[id as keyof typeof reveal] ? `Ẩn ${label}` : `Hiện ${label}`}
                >
                  {reveal[id as keyof typeof reveal] ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>
          ))}

        </div>

        <p className="settings-notice">
          <ShieldCheck size={16} aria-hidden="true" />
          <span>
            Khóa chỉ lưu trong trình duyệt này (<code>localStorage</code>) và gửi kèm yêu cầu dưới dạng HTTP header
            tới backend. Để trống để dùng khóa cấu hình trên máy chủ.
          </span>
        </p>

        <div className="modal-actions">
          {saved && <span className="save-toast"><Check size={14} /> Đã lưu</span>}
          <button className="btn-cancel" onClick={() => ref.current?.close()}>Hủy</button>
          <button className="btn-save" onClick={handleSave}>Lưu khóa</button>
        </div>
      </div>
    </dialog>
  );
}

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
  source?: "llm" | "fallback" | "deterministic" | "bot_info" | "off_topic";
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
  top_label?: string | null;
  col_map: Record<string, string>;
  unmapped_cols: string[];
  is_ecommerce: boolean;
  suggested_queries: string[];
}

interface SheetInfo {
  file_name: string | null;
  name: string;
  rows: number;
  columns: number;
  column_names: string[];
}
interface SheetRelationship {
  sheet1: string;
  sheet2: string;
  join_key: string | null;
  relationship_type: string;
  similarity_score: number;
}

// ── API ───────────────────────────────────────────────────────────────────────
function makeApi(keys: ReturnType<typeof loadKeys>) {
  const kh = keyHeaders(keys);

  return {
    async upload(files: File[]) {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      // FormData: can't set Content-Type (browser sets it with boundary), add key headers separately
      const r = await fetch("/api/upload", { method: "POST", headers: kh, body: form });
      if (!r.ok) throw new Error(await errorDetail(r, "Tải tệp lên thất bại"));
      return r.json();
    },
    async importUrl(url: string) {
      const r = await fetch("/api/import-url", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...kh },
        body: JSON.stringify({ url }),
      });
      if (!r.ok) throw new Error(await errorDetail(r, "Nhập từ URL thất bại"));
      return r.json();
    },
    async sampleShop() {
      const r = await fetch("/api/sample-shop", { method: "POST", headers: kh });
      if (!r.ok) throw new Error(await errorDetail(r, "Không nạp được shop mẫu"));
      return r.json();
    },
    keyHeaders: kh,
  };
}

// ── Simple markdown renderer ──────────────────────────────────────────────────
// Inline **bold** and `code` — the backend appends "**Nên kiểm tra:**", which used to render as raw asterisks.
function inline(text: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
    part.length > 4 && part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong>
    : part.length > 2 && part.startsWith("`") && part.endsWith("`") ? <code key={i}>{part.slice(1, -1)}</code>
    : part);
}

const NUMERIC_CELL = /^[-−+]?\d[\d.,]*\s*(%|đ|₫|tr|tỷ)?$/;

function MdTable({ rows }: { rows: string[][] }) {
  const [head, ...body] = rows;
  const cell = (c: string) => (NUMERIC_CELL.test(c) ? "num" : undefined);
  return (
    <div className="md-table">
      <table>
        <thead><tr>{head.map((c, j) => <th key={j}>{inline(c)}</th>)}</tr></thead>
        <tbody>{body.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className={cell(c)}>{inline(c)}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

// ponytail: line-based markdown subset (headings, lists, bold, code, pipe tables) — swap for react-markdown if answers grow richer.
function MdText({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trimStart().startsWith("|")) {
      const rows: string[][] = [];
      for (; i < lines.length && lines[i].trimStart().startsWith("|"); i++) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells); // drop the |---|---| separator
      }
      i--;
      if (rows.length) out.push(<MdTable key={i} rows={rows} />);
      continue;
    }
    if (line.startsWith("### ")) out.push(<h3 key={i}>{inline(line.slice(4))}</h3>);
    else if (line.startsWith("## ")) out.push(<h2 key={i}>{inline(line.slice(3))}</h2>);
    else if (line.startsWith("# ")) out.push(<h1 key={i}>{inline(line.slice(2))}</h1>);
    else if (line.startsWith("⚠️")) out.push(<p key={i} className="warn">{inline(line)}</p>);
    else if (/^\d+\.\s/.test(line)) out.push(<p key={i} className="li">{inline(line)}</p>);
    else if (line.startsWith("- ") || line.startsWith("* ")) out.push(<p key={i} className="li">{inline(line.slice(2))}</p>);
    else if (line.trim() === "") out.push(<div key={i} className="br" />);
    else out.push(<p key={i}>{inline(line)}</p>);
  }
  return <div className="md">{out}</div>;
}

// ── Source badge ──────────────────────────────────────────────────────────────
const DIRECT = { label: "Tính trực tiếp từ dữ liệu", title: "Câu trả lời soạn thẳng từ kết quả tính, không qua AI", cls: "badge-fallback" };
const SOURCE_LABELS: Record<string, { label: string; title: string; cls: string }> = {
  llm:           { label: "AI tóm tắt từ số đã tính", title: "AI viết lại kết quả đã tính; con số nào không khớp dữ liệu sẽ bị loại", cls: "badge-llm" },
  fallback:      DIRECT,
  deterministic: DIRECT,
  bot_info:      { label: "Giới thiệu", title: "Giới thiệu về trợ lý, không phải phân tích dữ liệu", cls: "badge-bot" },
  off_topic:     { label: "Ngoài phạm vi", title: "Câu hỏi nằm ngoài dữ liệu đã tải lên", cls: "badge-bot" },
};

function SourceBadge({ source }: { source: string }) {
  const info = SOURCE_LABELS[source] ?? { label: source, title: source, cls: "badge-bot" };
  return <span className={`source-badge ${info.cls}`} title={info.title}>{info.label}</span>;
}

// ── Plan detail — human-readable execution plan ───────────────────────────────
const AGG_LABELS: Record<string, string> = {
  sum: "tổng", mean: "trung bình", median: "trung vị", count: "đếm", nunique: "đếm khác nhau", min: "nhỏ nhất", max: "lớn nhất",
};

function PlanDetail({ queries }: { queries: string[] }) {
  if (!queries || queries.length === 0) return null;

  type Obj = Record<string, unknown>;
  const renderPlan = (raw: string) => {
    try {
      const p = JSON.parse(raw);
      const chips: string[] = [];
      // Current planner emits metrics[{column, aggregation}] + sort[{column, direction}]; older plans used metric_col/agg.
      const metrics: Obj[] = p.metrics?.length ? p.metrics : p.metric_col ? [{ column: p.metric_col, aggregation: p.agg }] : [];
      metrics.forEach((m) => chips.push(`${AGG_LABELS[String(m.aggregation)] ?? m.aggregation ?? ""} ${m.column}`.trim()));
      if (p.group_by?.length) chips.push(`theo ${p.group_by.join(", ")}`);
      if (p.filters?.length) chips.push(`lọc ${p.filters.map((f: Obj) => `${f.column} = ${Array.isArray(f.value) ? f.value.join("/") : f.value}`).join(", ")}`);
      const sort = Array.isArray(p.sort) ? p.sort[0] : null;
      if (sort) chips.push(`sắp xếp ${sort.direction === "asc" ? "tăng" : "giảm"} dần`);
      if (p.limit) chips.push(`lấy ${p.limit} dòng đầu`);
      if (p.derived_columns?.length) chips.push(`cột tính thêm: ${p.derived_columns.map((d: Obj) => d.name).join(", ")}`);
      if (!chips.length && p.action) chips.push(String(p.action));
      return chips.length ? chips : [raw];
    } catch {
      return [raw];
    }
  };

  return (
    <details className="plan-detail">
      <summary>Cách tính</summary>
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
  classify:   "Hiểu câu hỏi",
  planner:    "Lập kế hoạch",
  execute:    "Tính toán",
  synthesize: "Viết trả lời",
  bot_info:   "Giới thiệu",
  off_topic:  "Ngoài phạm vi",
  data_summary: "Tóm tắt dữ liệu",
  needs_cogs: "Kiểm tra giá vốn",
};

const PIPELINE = ["classify", "planner", "execute", "synthesize"];

// Full pipeline as a stepper: done ✓ / running ● / pending ○. Short-circuit routes
// (bot_info, off_topic) never enter the pipeline, so they show only what ran.
function NodeProgress({ nodes, streaming }: { nodes: string[]; streaming?: boolean }) {
  if (!nodes.length) return null;
  const steps = nodes.some((n) => !PIPELINE.includes(n)) ? nodes : PIPELINE;
  const firstPending = steps.findIndex((n) => !nodes.includes(n));
  return (
    <ol className="node-progress" aria-label="Tiến trình agent">
      {steps.map((n, i) => {
        const state = nodes.includes(n) ? "done" : streaming && i === firstPending ? "running" : "pending";
        return (
          <li key={n} className={`node-step ${state}`}>
            {state === "done" ? <Check size={12} strokeWidth={3} aria-hidden="true" /> : <span className="node-dot" aria-hidden="true" />}
            {NODE_LABELS[n] ?? n}
          </li>
        );
      })}
    </ol>
  );
}

// ── Agent Steps display ───────────────────────────────────────────────────────
function AgentStepsPanel({ steps }: { steps: AgentStep[] }) {
  if (!steps.length) return null;
  return (
    <details className="agent-steps">
      <summary>Agent thực hiện {steps.length} bước</summary>
      <div className="agent-steps-list">
        {steps.map((s) => (
          <div key={s.step} className="agent-step-item">
            <span className="agent-step-icon">{s.step}</span>
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
        // Colour says good/bad, the word says it too — "▲ phí sàn" is bad news in green otherwise.
        <div className={`kpi-delta ${card.delta_positive ? "positive" : "negative"}`}>
          {card.delta} · {card.delta_positive ? "tốt hơn" : "xấu hơn"} kỳ trước
        </div>
      )}
      {card.formula && <div className="kpi-formula" title={card.formula}>{card.formula}</div>}
    </div>
  );
}

// Workspace charts share one card shape; Plotly colours are tuned for the dark theme.
// memo + useMemo: chat re-renders on every streamed token — without them every chart in the
// history got a fresh layout object and Plotly redrew all of them per token.
const PLOT_CONFIG = { displayModeBar: false };
const ChartCard = memo(function ChartCard({ chart, height, exportHref }: { chart: ChartSpec; height: number; exportHref?: string }) {
  const layout = useMemo(() => {
    const base = chart.plotly_json.layout as Record<string, object>;
    return {
      ...base,
      title: undefined, // the card's figcaption already shows it — Plotly drew it a second time
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#a9b0d6", family: "Be Vietnam Pro, sans-serif" },
      colorway: ["#5eead4", "#a78bfa", "#f6c177", "#56609e", "#99f6e4"],
      separators: ",.", // Vietnamese: 1.234,5
      xaxis: { automargin: true, gridcolor: "rgba(129,140,248,.14)", zerolinecolor: "rgba(129,140,248,.3)", ...(base.xaxis ?? {}) },
      yaxis: { automargin: true, gridcolor: "rgba(129,140,248,.14)", zerolinecolor: "rgba(129,140,248,.3)", ...(base.yaxis ?? {}) },
      margin: { l: 56, r: 16, t: 24, b: 56 },
    };
  }, [chart]);
  const style = useMemo(() => ({ width: "100%", height: `${height}px` }), [height]);
  return (
    <figure className="chart-card">
      <div className="chart-head">
        <figcaption className="chart-title">{chart.title}</figcaption>
        {exportHref && <a className="chart-export-link" href={exportHref} download>Tải PNG</a>}
      </div>
      <Suspense fallback={<div className="chart-loading" style={style} aria-label="Đang tải biểu đồ" />}>
        <Plot data={chart.plotly_json.data as never} layout={layout} useResizeHandler style={style} config={PLOT_CONFIG} />
      </Suspense>
    </figure>
  );
});

// ── Dashboard Panel ───────────────────────────────────────────────────────────
function DashboardPanel({
  data,
  sessionId,
  sheetName,
  onAsk,
}: {
  data: DashboardData;
  sessionId: string;
  sheetName: string;
  onAsk: (q: string) => void;
}) {
  const [subTab, setSubTab] = React.useState<"kpi" | "products" | "trends">("kpi");
  const trendChart = data.charts.find((c) => c.chart_type === "line");
  const otherCharts = data.charts.filter((c) => c.chart_type !== "line");
  const chartPng = (id: string) => `/api/dashboard/${sessionId}/export-chart/${id}.png`;

  return (
    <div className="panel dashboard-panel">
      <div className="panel-title-row">
        <div className="panel-title-block">
          <h2 className="panel-head">Tổng quan{sheetName ? ` ${sheetName}` : ""}</h2>
          <p className="panel-sub">KPI do agent chọn theo miền dữ liệu đã nhận diện</p>
        </div>
        {data.platform && (
          <span className="platform-badge">{data.platform}</span>
        )}
      </div>

      <div className="dashboard-toolbar">
        <div className="segmented" role="group" aria-label="Chế độ xem">
          {(["kpi", "products", "trends"] as const).map((t) => (
            <button
              key={t}
              className={`dash-tab${subTab === t ? " active" : ""}`}
              aria-pressed={subTab === t}
              onClick={() => setSubTab(t)}
            >
              {t === "kpi" && "KPI"}
              {t === "products" && "Top 10"}
              {t === "trends" && "Xu hướng"}
            </button>
          ))}
        </div>
        {data.unmapped_cols.length > 0 && (
          <p className="unmapped-notice" title="Hỏi chatbot để phân tích các cột này thủ công">
            <CircleAlert size={14} aria-hidden="true" />
            Chưa nhận diện cột <span className="mono">{data.unmapped_cols.join(", ")}</span>
          </p>
        )}
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
            <div className="chart-grid">
              {otherCharts.map((c) => (
                <ChartCard key={c.chart_id} chart={c} height={300} exportHref={chartPng(c.chart_id)} />
              ))}
            </div>
          )}
          {/* AI-generated suggested queries */}
          {data.suggested_queries?.length > 0 && (
            <div className="suggestions">
              <div className="sugg-label">Phân tích sâu hơn</div>
              <div className="chip-wrap">
                {data.suggested_queries.map((q) => (
                  <button key={q} className="chip" onClick={() => onAsk(q)}>{q}</button>
                ))}
              </div>
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
                  <th className="num">#</th>
                  <th>Nhóm</th>
                  <th className="num">{data.top_label || "Giá trị"}</th>
                </tr>
              </thead>
              <tbody>
                {data.top_products.map((p, i) => (
                  <tr key={i}>
                    <td className="num">{String(p.rank ?? i + 1)}</td>
                    <td>{String(p.name ?? "")}</td>
                    <td className="num">{typeof p.value === "number" ? Math.round(p.value).toLocaleString("vi-VN") : String(p.value ?? "—")}</td>
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
            <ChartCard chart={trendChart} height={360} exportHref={chartPng(trendChart.chart_id)} />
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

// ── Sheets panel — view + switch + merge sheets/files ─────────────────────────
function SheetsPanel({
  sheets,
  relationships,
  activeSheet,
  sheetKey,
  onSwitch,
  onMerge,
  busy,
}: {
  sheets: SheetInfo[];
  relationships: SheetRelationship[];
  activeSheet: string | null;
  sheetKey: (s: SheetInfo) => string;
  onSwitch: (key: string) => void;
  onMerge: (names: string[], joinKey: string | null) => void;
  busy: boolean;
}) {
  const files = Array.from(new Set(sheets.map((s) => s.file_name).filter(Boolean)));
  const joinable = relationships.filter((r) => r.join_key);
  return (
    <>
      <div className="sheets-panel">
        <div className="sheets-label">
          <FileSpreadsheet size={16} aria-hidden="true" />
          <span className="sheets-file">{files.length === 1 ? files[0] : `${files.length || sheets.length} tệp`}</span>
          <span className="sheets-count">{sheets.length} sheet</span>
        </div>
        <div className="sheets-list">
          {sheets.map((s) => {
            const key = sheetKey(s);
            const active = key === activeSheet;
            return (
              <button
                key={key}
                className={`sheet-item${active ? " active" : ""}`}
                aria-current={active || undefined}
                disabled={busy}
                onClick={() => onSwitch(key)}
                title={s.column_names.join(", ")}
              >
                <span className="sheet-dot" aria-hidden="true" />
                <span className="sheet-name">{s.name}</span>
                <span className="sheet-meta">{s.rows.toLocaleString("vi-VN")}×{s.columns}</span>
              </button>
            );
          })}
        </div>
      </div>
      {joinable.length > 0 && (
        <div className="sheets-rels">
          <div className="sheets-rel-label">Có thể gộp</div>
          {joinable.map((r, i) => (
            <div key={i} className="sheet-rel">
              <div className="sheet-rel-text">
                <span>{r.sheet1.split("::").pop()} ↔ {r.sheet2.split("::").pop()}</span>
                <span className="sheet-rel-key">qua {r.join_key}</span>
              </div>
              <button
                className="sheet-merge-btn"
                disabled={busy}
                onClick={() => onMerge([r.sheet1, r.sheet2], r.join_key)}
              >
                <GitMerge size={14} aria-hidden="true" /> Gộp
              </button>
            </div>
          ))}
        </div>
      )}
    </>
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
  // Chat is a permanent column now — `tab` only drives the workspace column.
  const [tab, setTab] = useState<"dashboard" | "preview" | "charts">("preview");
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [allCharts, setAllCharts] = useState<ChartSpec[]>([]);
  const [reportId, setReportId] = useState("");
  const [sheets, setSheets] = useState<SheetInfo[]>([]);
  const [relationships, setRelationships] = useState<SheetRelationship[]>([]);
  const [activeSheet, setActiveSheet] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [importUrl, setImportUrl] = useState("");
  const [showImport, setShowImport] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [apiKeys, setApiKeys] = useState(loadKeys);
  const [dataNotes, setDataNotes] = useState<string[]>([]);
  const [cogsMissing, setCogsMissing] = useState(0);
  const [toast, setToast] = useState("");
  const [askAgain, setAskAgain] = useState<string[]>([]);   // questions from a session the server dropped
  const [navOpen, setNavOpen] = useState(false);            // phone: sidebar drawer
  const [pane, setPane] = useState<"chat" | "workspace">("chat"); // < 1180px: one column at a time
  const fileRef = useRef<HTMLInputElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);                            // user is at the bottom of the chat
  const abortRef = useRef<AbortController | null>(null);

  // Rebuild only when keys change — send() closes over this via useCallback deps,
  // so a stale `api` would keep sending the old API-key headers after Settings save.
  const api = useMemo(() => makeApi(apiKeys), [apiKeys]);
  const hasKey = !!(apiKeys.gemini || apiKeys.anthropic);

  // Follow the stream only while the user is at the bottom — reading an older answer must not get yanked down.
  useEffect(() => {
    const el = messagesRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 6000);
    return () => clearTimeout(t);
  }, [toast]);

  useEffect(() => {
    if (!navOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setNavOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navOpen]);

  // Reconstruct the storage key ("file::sheet") from a SheetInfo.
  const sheetKey = (s: SheetInfo) =>
    s.file_name && s.file_name !== s.name ? `${s.file_name}::${s.name}` : s.name;

  function fetchDashboard(sid: string) {
    fetch(`/api/dashboard/${sid}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((dash: DashboardData | null) => {
        if (!dash || !dash.kpi_cards?.length) return;
        setDashboardData(dash);
      })
      .catch(() => {});
  }

  // Session routes only 404 when the session is gone — on Render free tier ./data is
  // wiped on every restart/redeploy. Drop back to the upload screen instead of a dead UI.
  function expireSession() {
    abortRef.current?.abort();
    setToast("Phiên làm việc đã hết hạn (máy chủ vừa khởi động lại). Tải lại file để hỏi tiếp — các câu đã hỏi được giữ lại bên dưới ô chat.");
    setAskAgain(messages.filter((m) => m.role === "user").map((m) => m.content));
    setDataNotes([]);
    setCogsMissing(0);
    setSessionId("");
    setProfile(null);
    setPreviewCols([]);
    setPreviewRows([]);
    setSuggestions([]);
    setMessages([]);
    setAllCharts([]);
    setDashboardData(null);
    setReportId("");
    setSheets([]);
    setRelationships([]);
    setActiveSheet(null);
  }

  function applyUploadResponse(d: Record<string, unknown>) {
    const sid = d.session_id as string;
    setSessionId(sid);
    setProfile(d.profile as Profile);
    setPreviewCols((d.preview_columns as string[]) ?? []);
    setPreviewRows((d.preview_rows as Record<string, string>[]) ?? []);
    setSuggestions((d.suggested_queries as string[]) ?? []);
    setActiveSheet((d.active_sheet as string) ?? null);
    setDataNotes((d.data_notes as string[]) ?? []);
    setCogsMissing((d.cogs_missing as number) ?? 0);
    setMessages([]);
    setAllCharts([]);
    setDashboardData(null);
    setSheets([]);
    setRelationships([]);
    setTab("preview");
    setShowImport(false);
    setNavOpen(false);

    // Multi-sheet / multi-file workbook → load the sheet inventory so the user can
    // see + switch which sheet is analyzed (no more silent single-sheet selection).
    const sheetNames = (d.sheet_names as string[]) ?? [];
    if (sheetNames.length > 1) {
      fetch(`/api/sheets/${sid}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((s) => {
          if (!s) return;
          setSheets((s.sheets as SheetInfo[]) ?? []);
          setRelationships((s.relationships as SheetRelationship[]) ?? []);
        })
        .catch(() => {});
    }

    fetchDashboard(sid);
  }

  // Apply an ActiveSheetResponse / MergeSheetsResponse refresh payload.
  // Switching sheet / merging = a different DataFrame, so chat history, charts and
  // the report from the previous frame no longer apply — clear them.
  function applyRefresh(d: Record<string, unknown>) {
    setProfile(d.profile as Profile);
    setPreviewCols((d.preview_columns as string[]) ?? []);
    setPreviewRows((d.preview_rows as Record<string, string>[]) ?? []);
    if (d.suggested_queries) setSuggestions(d.suggested_queries as string[]);
    if ("data_notes" in d) setDataNotes((d.data_notes as string[]) ?? []);
    if ("cogs_missing" in d) setCogsMissing((d.cogs_missing as number) ?? 0);
    setActiveSheet((d.active_sheet as string) ?? null);
    setMessages([]);
    setAllCharts([]);
    setReportId("");
    setDashboardData(null);          // force dashboard recompute for the new frame
    setNavOpen(false);
    fetchDashboard(sessionId);
  }

  async function switchSheet(key: string) {
    if (!sessionId || key === activeSheet) return;
    setBusy(true);
    try {
      const r = await fetch(`/api/session/${sessionId}/active-sheet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sheet_name: key }),
      });
      if (r.status === 404) return expireSession();
      if (!r.ok) throw new Error(await errorDetail(r, "Đổi sheet thất bại"));
      applyRefresh(await r.json());
      setTab("preview");
    } catch (e: unknown) {
      setToast((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function mergeSheets(names: string[], joinKey: string | null) {
    if (!sessionId || names.length < 2) return;
    setBusy(true);
    try {
      const r = await fetch(`/api/merge-sheets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, sheet_names: names, join_key: joinKey || undefined }),
      });
      if (r.status === 404) return expireSession();
      if (!r.ok) throw new Error(await errorDetail(r, "Gộp sheet thất bại"));
      const resp = await r.json();
      applyRefresh(resp);
      // Refresh sheet inventory so the new merged sheet shows up.
      fetch(`/api/sheets/${sessionId}`)
        .then((x) => (x.ok ? x.json() : null))
        .then((s) => { if (s) { setSheets(s.sheets ?? []); setRelationships(s.relationships ?? []); } })
        .catch(() => {});
      setTab("preview");
    } catch (e: unknown) {
      setToast((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleFiles(files: FileList | File[]) {
    const arr = Array.from(files).filter((f) =>
      [".csv", ".xlsx", ".xls"].some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (!arr.length) return setToast("Chỉ nhận tệp CSV, XLSX hoặc XLS.");
    setBusy(true);
    try {
      applyUploadResponse(await api.upload(arr));
    } catch (e: unknown) {
      setToast((e as Error).message);
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
      setToast((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function loadSampleShop() {
    setBusy(true);
    try {
      applyUploadResponse(await api.sampleShop());
    } catch (e: unknown) {
      setToast((e as Error).message);
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
    setPane("chat");
    stickRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;

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
        headers: { "Content-Type": "application/json", ...api.keyHeaders },
        body: JSON.stringify({ session_id: sessionId, question: q }),
        signal: controller.signal,
      });

      if (resp.status === 404) return expireSession();
      if (!resp.ok) throw new Error(await errorDetail(resp, "Không nhận được câu trả lời"));

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
      const stopped = (e as Error).name === "AbortError";
      setMessages((m) => m.map((msg, i) =>
        i === assistantIdx
          ? { ...msg, streaming: false, content: stopped ? `${msg.content}\n\n(Đã dừng)`.trim() :`❌ ${(e as Error).message}` }
          : msg
      ));
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }, [sessionId, busy, api, messages]);

  const numericCount = profile ? Object.keys(profile.numeric_summary).length : 0;
  const missingCount = profile ? Object.values(profile.missing_values).reduce((a, b) => a + b, 0) : 0;
  const sheetName = activeSheet === "__concat__"
    ? `gộp ${sheets.length} sheet`
    : (activeSheet?.split("::").pop() ?? "");
  const hasDashboard = !!dashboardData?.kpi_cards?.length;
  const streaming = !!messages[messages.length - 1]?.streaming;
  const lastAnswer = [...messages].reverse().find((m) => m.role === "assistant" && !m.streaming)?.content ?? "";
  const numCols = new Set(Object.entries(profile?.column_types ?? {})
    .filter(([, t]) => /^(int|float)/.test(t)).map(([c]) => c));

  const dropHandlers = {
    onDragOver: (e: React.DragEvent) => { e.preventDefault(); setDragging(true); },
    onDragLeave: () => setDragging(false),
    onDrop: (e: React.DragEvent) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); },
  };

  const urlImport = (compact: boolean) => (
    <div className={`url-import${compact ? " compact" : ""}`}>
      <label htmlFor={compact ? "import-url-rail" : "import-url"}>{compact ? "Nhập từ URL" : "Hoặc nhập từ URL"}</label>
      <div className="url-row">
        <input id={compact ? "import-url-rail" : "import-url"} className="import-input" type="url"
          placeholder="https://docs.google.com/spreadsheets/…"
          value={importUrl} onChange={(e) => setImportUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleImportUrl()} />
        <button className="import-btn" disabled={!importUrl.trim() || busy} onClick={handleImportUrl}>
          {busy ? "Đang tải…" : "Nhập"}
        </button>
      </div>
      <span className="import-hint">CSV/XLSX công khai, Google Sheets chia sẻ “bất kỳ ai có đường liên kết”, liên kết Dropbox</span>
    </div>
  );

  // Starfield + planet sit behind the welcome screen only; data screens stay calm.
  const planet = (
    <svg className="welcome-planet" viewBox="0 0 440 320" aria-hidden="true">
      <defs>
        <linearGradient id="planet-g" x1="150" y1="40" x2="330" y2="240" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#a78bfa" />
          <stop offset=".55" stopColor="#4c3fb8" />
          <stop offset="1" stopColor="#0e1240" />
        </linearGradient>
      </defs>
      <circle cx="240" cy="150" r="92" fill="url(#planet-g)" />
      <ellipse cx="240" cy="150" rx="170" ry="30" fill="none" stroke="#c4b5fd" strokeOpacity=".45" strokeWidth="2" transform="rotate(-18 240 150)" />
      <ellipse cx="240" cy="150" rx="200" ry="38" fill="none" stroke="#5eead4" strokeOpacity=".2" transform="rotate(-18 240 150)" />
    </svg>
  );

  return (
    <div className="layout">
      {/* Settings modal */}
      {showSettings && (
        <SettingsModal
          onClose={() => {
            setApiKeys(loadKeys()); // re-read in case saved
            setShowSettings(false);
          }}
        />
      )}
      <input ref={fileRef} type="file" multiple accept=".csv,.xlsx,.xls" hidden
        onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.target.value = ""; }} />

      {toast && (
        <div className="toast" role="alert">
          <CircleAlert size={16} aria-hidden="true" />
          <span>{toast}</span>
          <button className="icon-btn" onClick={() => setToast("")} aria-label="Đóng thông báo"><X size={14} /></button>
        </div>
      )}

      {/* Phone only: the sidebar becomes a drawer behind this bar (it used to be display:none — no way to add files or switch sheets) */}
      <header className="topbar">
        <button className="icon-btn" onClick={() => setNavOpen(true)} aria-label="Mở menu" aria-expanded={navOpen} aria-controls="sidebar">
          <Menu size={20} />
        </button>
        <span className="topbar-brand">SellerLens</span>
      </header>
      {navOpen && <div className="scrim" onClick={() => setNavOpen(false)} aria-hidden="true" />}

      {/* Sidebar */}
      <aside id="sidebar" className={`sidebar${navOpen ? " open" : ""}`}>
        <div className="brand">
          <svg width="30" height="30" viewBox="0 0 28 28" fill="none" aria-hidden="true">
            <defs>
              <linearGradient id="brand-g" x1="6" y1="6" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#a78bfa"/>
                <stop offset="100%" stopColor="#5eead4"/>
              </linearGradient>
            </defs>
            <circle cx="14" cy="14" r="8" fill="url(#brand-g)"/>
            <ellipse cx="14" cy="14" rx="13" ry="4.5" stroke="#edefff" strokeOpacity=".85" strokeWidth="1.4" transform="rotate(-20 14 14)"/>
            <circle cx="24" cy="5" r="1.2" fill="#edefff"/>
          </svg>
          <span>SellerLens</span>
          <button className="icon-btn nav-close" onClick={() => setNavOpen(false)} aria-label="Đóng menu"><X size={18} /></button>
        </div>

        <section className="rail-section">
          <h2 className="rail-label">Nguồn dữ liệu</h2>
          {!sessionId ? (
            <p className="rail-empty">Chưa có dữ liệu. Tệp và sheet bạn tải lên sẽ hiện ở đây.</p>
          ) : (
            <>
              <button className="rail-add" aria-expanded={showImport} onClick={() => setShowImport((v) => !v)}>
                <Plus size={16} aria-hidden="true" /> Thêm tệp hoặc URL
              </button>
              {showImport && (
                <div className="rail-import">
                  <div className={`drop-zone${dragging ? " dragging" : ""}`} {...dropHandlers}>
                    <span className="drop-icon"><Upload size={18} /></span>
                    <span>Kéo thả tệp vào đây</span>
                    <button className="btn-link" onClick={() => fileRef.current?.click()} disabled={busy}>
                      {busy ? "Đang tải lên…" : "Chọn tệp"}
                    </button>
                  </div>
                  {urlImport(true)}
                </div>
              )}
              {sheets.length > 1 && (
                <SheetsPanel
                  sheets={sheets}
                  relationships={relationships}
                  activeSheet={activeSheet}
                  sheetKey={sheetKey}
                  onSwitch={switchSheet}
                  onMerge={mergeSheets}
                  busy={busy}
                />
              )}
            </>
          )}
        </section>

        {profile && (
          <section className="rail-section">
            <h2 className="rail-label">{sheets.length > 1 ? "Sheet đang phân tích" : "Dữ liệu đang phân tích"}</h2>
            <div className="stat-grid">
              <div className="stat"><span>Dòng</span><b>{profile.rows.toLocaleString("vi-VN")}</b></div>
              <div className="stat"><span>Cột</span><b>{profile.columns}</b></div>
              <div className="stat"><span>Cột số</span><b>{numericCount}</b></div>
              <div className="stat"><span>Ô thiếu</span><b className={missingCount ? "warn" : ""}>{missingCount.toLocaleString("vi-VN")}</b></div>
            </div>
          </section>
        )}

        <div className="sidebar-footer">
          {sessionId && (
            <a className="report-link" href={`/api/session/${sessionId}/data.csv`} download>
              <Download size={16} aria-hidden="true" /> Tải CSV đã xử lý
            </a>
          )}
          {reportId && (
            <a className="report-link" href={`/api/report/${reportId}`} target="_blank" rel="noreferrer">
              <FileText size={16} aria-hidden="true" /> Báo cáo Markdown
            </a>
          )}
          <button className="btn-settings" onClick={() => setShowSettings(true)}>
            <KeyRound size={16} aria-hidden="true" />
            <span className="btn-settings-label">Khóa API</span>
            <span className="btn-settings-status">{hasKey ? "Khóa riêng" : "Máy chủ"}</span>
            <span className={`key-indicator${hasKey ? " active" : ""}`} aria-hidden="true" />
          </button>
        </div>
      </aside>

      {!sessionId ? (
        /* ── Welcome — nothing uploaded yet ── */
        <main className="welcome">
          {planet}
          <div className="welcome-inner">
            <div className="welcome-intro">
              <span className="eyebrow">Cho người bán Shopee &amp; TikTok Shop</span>
              <h1>Biết lãi thật <span className="grad-text">sau phí sàn.</span></h1>
              <p>Kéo file export đơn hàng Shopee/TikTok vào đây, thêm bảng giá vốn nếu có. Hỏi như nói chuyện —
                mọi con số trong câu trả lời đều tính từ chính file của bạn.</p>
            </div>

            <div className={`drop-hero${dragging ? " dragging" : ""}`} {...dropHandlers}>
              <div className="drop-hero-icon"><Upload size={24} aria-hidden="true" /></div>
              <div className="drop-hero-text">
                <strong>Kéo file export Shopee/TikTok vào đây</strong>
                <span>Chọn nhiều file cùng lúc: đơn hàng, giá vốn (sku, gia_von), chi phí quảng cáo · cũng nhận CSV/Excel bất kỳ</span>
              </div>
              <div className="drop-hero-actions">
                <button className="btn-accent" onClick={() => fileRef.current?.click()} disabled={busy}>
                  {busy ? "Đang tải lên…" : "Chọn tệp"}
                </button>
                <button className="btn-ghost" onClick={loadSampleShop} disabled={busy}>
                  <Store size={16} aria-hidden="true" /> Dùng thử với shop mẫu
                </button>
              </div>
            </div>

            <ol className="how-steps" aria-label="Bạn có thể hỏi">
              <li><span className="how-num">Lãi từng SKU</span>“SKU nào bán chạy mà đang lỗ?”</li>
              <li><span className="how-num violet">Vì sao lãi đổi</span>“Vì sao lãi tháng này giảm?”</li>
              <li><span className="how-num amber">Phí sàn từng kênh</span>“Phí sàn Shopee và TikTok bên nào cao hơn?”</li>
            </ol>

            {urlImport(false)}

            <details className="trust">
              <summary>Vì sao tin được con số?</summary>
              <p>AI chỉ chọn cách tính, và cách tính được kiểm tra với cột thật trong file. Phép tính chạy bằng
                code cố định, không chạy code tự do. Câu trả lời có con số không khớp kết quả tính sẽ bị loại.
                Shop mẫu là dữ liệu mô phỏng.</p>
            </details>
          </div>
        </main>
      ) : (
      /* Main — chat column (always visible) + workspace column */
      <main className="main" data-pane={pane}>
        {/* < 1180px: chat and workspace take turns instead of splitting the height 50/50 */}
        <div className="pane-switch segmented" role="group" aria-label="Chế độ xem">
          <button className={`dash-tab${pane === "chat" ? " active" : ""}`} aria-pressed={pane === "chat"} onClick={() => setPane("chat")}>
            <MessageSquare size={15} aria-hidden="true" /> Trò chuyện
          </button>
          <button className={`dash-tab${pane === "workspace" ? " active" : ""}`} aria-pressed={pane === "workspace"} onClick={() => setPane("workspace")}>
            <LayoutDashboard size={15} aria-hidden="true" /> Số liệu
            {allCharts.length > 0 && <span className="badge">{allCharts.length}</span>}
          </button>
        </div>

        {/* Data gaps that make profit numbers wrong stay pinned — not buried in one chat answer */}
        {dataNotes.length > 0 && (
          <div className="data-notice" role="status">
            <TriangleAlert size={18} aria-hidden="true" />
            <ul>{dataNotes.map((n) => <li key={n}>{n}</li>)}</ul>
            {cogsMissing > 0 && (
              <a className="notice-btn" href={`/api/session/${sessionId}/cogs-template.csv`} download>
                <Download size={14} aria-hidden="true" /> Tải mẫu giá vốn
              </a>
            )}
          </div>
        )}

        {/* ── Chat column — permanent, no longer a tab ── */}
        <section className="chat-col">
          <div className="col-head">
            <h2 className="col-title">Trò chuyện</h2>
            {sheetName && <span className="col-sub">với sheet {sheetName}</span>}
            {busy && <span className="col-status">Đang xử lý…</span>}
          </div>

          {/* Streamed text changes per token — announce only the finished answer to screen readers */}
          <div className="sr-only" aria-live="polite">{lastAnswer}</div>
          <div className="messages" ref={messagesRef}
            onScroll={(e) => { const el = e.currentTarget; stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>
            {messages.length === 0 && (
              <div className="empty">
                <div className="empty-icon"><MessageSquare size={32} strokeWidth={1.5} /></div>
                <div className="empty-title">Hỏi gì về dữ liệu cũng được</div>
                <div className="empty-sub">
                  {suggestions.length ? "Bấm một câu gợi ý bên dưới, hoặc tự gõ câu hỏi" : "vd. “lãi tháng này bao nhiêu?” · “SKU nào đang lỗ?”"}
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`msg ${msg.role}`}>
                {msg.role === "user" ? (
                  <div className="bubble">{msg.content}</div>
                ) : (
                  <>
                    <svg className="msg-avatar" width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
                      <circle cx="14" cy="14" r="8" fill="url(#brand-g)" />
                      <ellipse cx="14" cy="14" rx="13" ry="4.5" fill="none" stroke="#edefff" strokeOpacity=".85" strokeWidth="1.4" transform="rotate(-20 14 14)" />
                    </svg>
                    <div className="assistant-msg">
                      {msg.nodes && msg.nodes.length > 0 && (
                        <NodeProgress nodes={msg.nodes} streaming={msg.streaming} />
                      )}
                      {msg.content ? (
                        <div className="answer">
                          <MdText text={msg.content} />
                          {msg.streaming && <span className="cursor">▋</span>}
                        </div>
                      ) : msg.streaming ? (
                        <div className="skeleton" aria-label="Đang trả lời"><span /><span /></div>
                      ) : null}
                      {msg.charts && msg.charts.length > 0 && (
                        <div className="msg-charts">
                          {msg.charts.map((c) => <ChartCard key={c.chart_id} chart={c} height={260} />)}
                        </div>
                      )}
                      {msg.agentSteps && <AgentStepsPanel steps={msg.agentSteps} />}
                      {msg.queries && msg.queries.length > 0 && <PlanDetail queries={msg.queries} />}
                      {!msg.streaming && msg.source && (
                        <div className="msg-meta"><SourceBadge source={msg.source} /></div>
                      )}
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>

          <div className="composer">
            {askAgain.length > 0 && (
              <div className="chat-suggestions">
                <span className="sugg-label">Hỏi lại</span>
                {askAgain.map((s) => (
                  <button key={s} className="chip" disabled={busy}
                    onClick={() => { setAskAgain((a) => a.filter((x) => x !== s)); send(s); }}>{s}</button>
                ))}
              </div>
            )}
            {suggestions.length > 0 && (
              <div className="chat-suggestions">
                <span className="sugg-label">Gợi ý</span>
                {suggestions.map((s) => (
                  <button key={s} className="chip" disabled={busy} onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            )}
            <div className="input-bar">
              <label htmlFor="ask" className="sr-only">Câu hỏi về dữ liệu</label>
              {/* Stays editable while an answer streams — you can draft the next question */}
              <textarea id="ask" rows={2}
                placeholder="Hỏi về shop của bạn… Enter để gửi, Shift+Enter xuống dòng"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(question); } }}
              />
              <div className="input-btns">
                <span className="input-hint">Mọi con số được tính từ file của bạn</span>
                {streaming ? (
                  <button className="btn-stop" onClick={() => abortRef.current?.abort()}>
                    <Square size={13} aria-hidden="true" /> Dừng
                  </button>
                ) : (
                  <button className="btn-primary" disabled={!question.trim() || busy} onClick={() => send(question)}>
                    {busy ? "Đang xử lý…" : <><Send size={15} aria-hidden="true" /> Hỏi</>}
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ── Workspace column — dashboard / data / charts ── */}
        <section className="workspace">
          <div className="ws-tabs">
            <nav className="ws-tab-list" aria-label="Workspace">
              {hasDashboard && (
                <button className={`ws-tab${tab === "dashboard" ? " active" : ""}`}
                  aria-current={tab === "dashboard" ? "page" : undefined}
                  onClick={() => setTab("dashboard")}><LayoutDashboard size={15} aria-hidden="true" /> Tổng quan</button>
              )}
              <button className={`ws-tab${tab === "preview" ? " active" : ""}`}
                aria-current={tab === "preview" ? "page" : undefined}
                onClick={() => setTab("preview")}><Table2 size={15} aria-hidden="true" /> Dữ liệu</button>
              <button className={`ws-tab${tab === "charts" ? " active" : ""}`}
                aria-current={tab === "charts" ? "page" : undefined}
                onClick={() => setTab("charts")}>
                <LineChart size={15} aria-hidden="true" /> Biểu đồ
                {allCharts.length > 0 && <span className="badge">{allCharts.length}</span>}
              </button>
            </nav>
            {tab === "dashboard" && hasDashboard && (
              <a className="export-xlsx-btn" href={`/api/dashboard/${sessionId}/export.xlsx`} download>
                <Download size={14} aria-hidden="true" /> Xuất Excel
              </a>
            )}
            {tab === "preview" && (
              <a className="export-xlsx-btn" href={`/api/session/${sessionId}/data.csv`} download>
                <Download size={14} aria-hidden="true" /> Tải CSV
              </a>
            )}
          </div>

          <div className="ws-body">
            {/* Dashboard */}
            {tab === "dashboard" && (
              dashboardData ? (
                <DashboardPanel
                  data={dashboardData}
                  sessionId={sessionId}
                  sheetName={sheetName}
                  onAsk={(q) => send(q)}
                />
              ) : (
                <div className="panel">
                  <div className="empty">
                    <div className="empty-icon"><LayoutDashboard size={32} strokeWidth={1.5} /></div>
                    <div className="empty-sub">Chưa có dashboard cho dữ liệu hiện tại</div>
                  </div>
                </div>
              )
            )}

            {/* Data preview */}
            {tab === "preview" && (
              <div className="panel">
                <div className="panel-title-block">
                  <h2 className="panel-head">{sheetName || "Dữ liệu"}</h2>
                  <p className="panel-sub">
                    {profile?.rows.toLocaleString("vi-VN")} dòng · {profile?.columns} cột · xem trước {previewRows.length} dòng đầu
                  </p>
                </div>
                {sheets.length > 1 && (
                  <div className="active-sheet-banner">
                    <FileSpreadsheet size={16} aria-hidden="true" />
                    <span>Đang phân tích <b>{sheetName}</b> · 1 trong {sheets.length} sheet — đổi sheet ở cột trái</span>
                  </div>
                )}
                <div className="tbl-wrap">
                  <table>
                    <thead><tr>{previewCols.map((c) => <th key={c} className={numCols.has(c) ? "num" : undefined}>{c}</th>)}</tr></thead>
                    <tbody>
                      {previewRows.map((row, i) => (
                        <tr key={i}>{previewCols.map((c) => {
                          const v = row[c];
                          const empty = v === null || v === undefined || v === "";
                          return <td key={c} className={numCols.has(c) ? "num" : undefined}>{empty ? <span className="cell-empty">trống</span> : v}</td>;
                        })}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {profile?.column_types && (
                  <section className="col-profile">
                    <h3 className="section-title">Hồ sơ cột</h3>
                    <div className="col-types-grid">
                      {Object.entries(profile.column_types).map(([col, dtype]) => {
                        const missing = profile.missing_values?.[col] ?? 0;
                        const kind = dtype.startsWith("int") || dtype.startsWith("float") ? "num"
                          : dtype.startsWith("datetime") ? "date"
                          : "cat";
                        return (
                          <div key={col} className="col-type-row">
                            <span className="col-name" title={col}>{col}</span>
                            {missing > 0 && <span className="col-missing">{missing.toLocaleString("vi-VN")} thiếu</span>}
                            <span className={`col-dtype dtype-${kind}`} title={dtype}>
                              {kind === "num" ? "số" : kind === "date" ? "ngày" : "phân loại"}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </section>
                )}
              </div>
            )}

            {/* Charts */}
            {tab === "charts" && (
              <div className="panel">
                {allCharts.length === 0 ? (
                  <div className="empty">
                    <div className="empty-icon"><LineChart size={32} strokeWidth={1.5} /></div>
                    <div className="empty-title">Chưa có biểu đồ</div>
                    <div className="empty-sub">Đặt câu hỏi để agent sinh biểu đồ</div>
                  </div>
                ) : (
                  <>
                    <h2 className="panel-head">Biểu đồ</h2>
                    <div className="chart-grid">
                      {allCharts.map((c) => <ChartCard key={c.chart_id} chart={c} height={320} />)}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
