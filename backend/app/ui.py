def render_home() -> str:
    return """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data Analysis AI Agent</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #f8f9fb;
      --surface: #ffffff;
      --surface2: #f1f3f7;
      --border: #e4e7ed;
      --border2: #d0d5e0;
      --text: #111827;
      --text2: #6b7280;
      --text3: #9ca3af;
      --accent: #4f46e5;
      --accent-light: #eef2ff;
      --accent-text: #3730a3;
      --success: #059669;
      --success-bg: #ecfdf5;
      --danger: #dc2626;
      --danger-bg: #fef2f2;
      --radius: 10px;
      --radius-sm: 6px;
      --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }
    body { font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; min-height: 100vh; display: flex; flex-direction: column; }

    /* ── Topbar ── */
    .topbar { height: 52px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 20px; gap: 10px; position: sticky; top: 0; z-index: 100; }
    .topbar-logo { display: flex; align-items: center; gap: 8px; text-decoration: none; }
    .topbar-icon { width: 28px; height: 28px; background: var(--accent); border-radius: 7px; display: flex; align-items: center; justify-content: center; }
    .topbar-icon svg { width: 16px; height: 16px; fill: none; stroke: white; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .topbar-name { font-weight: 600; font-size: 14px; color: var(--text); }
    .topbar-badge { font-size: 11px; color: var(--text3); background: var(--surface2); border: 1px solid var(--border); padding: 2px 8px; border-radius: 20px; margin-left: 4px; }
    .topbar-status { margin-left: auto; display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text3); }
    .status-dot.ready { background: var(--success); box-shadow: 0 0 0 3px var(--success-bg); }

    /* ── Layout ── */
    .layout { display: flex; flex: 1; overflow: hidden; }

    /* ── Sidebar ── */
    .sidebar { width: 220px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 16px 12px; gap: 2px; flex-shrink: 0; }
    .sidebar-section { font-size: 10px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: 0.07em; padding: 8px 8px 4px; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: var(--radius-sm); cursor: pointer; color: var(--text2); font-size: 13px; transition: background 0.12s, color 0.12s; border: none; background: none; width: 100%; text-align: left; }
    .nav-item svg { width: 16px; height: 16px; flex-shrink: 0; stroke: currentColor; fill: none; stroke-width: 1.75; stroke-linecap: round; stroke-linejoin: round; }
    .nav-item:hover { background: var(--surface2); color: var(--text); }
    .nav-item.active { background: var(--accent-light); color: var(--accent-text); font-weight: 500; }
    .sidebar-divider { height: 1px; background: var(--border); margin: 8px 0; }
    .sidebar-file { padding: 8px 10px; border-radius: var(--radius-sm); background: var(--surface2); margin-top: 4px; }
    .sidebar-file-name { font-size: 12px; font-weight: 500; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .sidebar-file-meta { font-size: 11px; color: var(--text3); margin-top: 2px; }

    /* ── Main ── */
    .main { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }

    /* ── Upload zone ── */
    .upload-zone { border: 1.5px dashed var(--border2); border-radius: var(--radius); padding: 32px 20px; text-align: center; cursor: pointer; transition: border-color 0.15s, background 0.15s; background: var(--surface); }
    .upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: var(--accent-light); }
    .upload-zone-icon { width: 40px; height: 40px; background: var(--accent-light); border-radius: 10px; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px; }
    .upload-zone-icon svg { width: 22px; height: 22px; stroke: var(--accent); fill: none; stroke-width: 1.75; stroke-linecap: round; stroke-linejoin: round; }
    .upload-zone h3 { font-size: 14px; font-weight: 500; color: var(--text); margin-bottom: 4px; }
    .upload-zone p { font-size: 12px; color: var(--text3); }
    .upload-zone-btn { display: inline-flex; align-items: center; gap: 6px; margin-top: 14px; background: var(--accent); color: white; font-size: 13px; font-weight: 500; padding: 8px 18px; border-radius: var(--radius-sm); border: none; cursor: pointer; transition: opacity 0.15s; }
    .upload-zone-btn:hover { opacity: 0.88; }
    #file-input { display: none; }

    /* ── Stats bar ── */
    .stats-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; }
    .stat-label { font-size: 11px; color: var(--text3); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .stat-value { font-size: 22px; font-weight: 600; color: var(--text); line-height: 1.2; }
    .stat-sub { font-size: 11px; color: var(--text3); margin-top: 2px; }

    /* ── Panel ── */
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
    .panel-header { padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; }
    .panel-title { font-size: 13px; font-weight: 600; color: var(--text); }
    .panel-body { padding: 18px; }

    /* ── Question input ── */
    .question-row { display: flex; gap: 10px; align-items: flex-start; }
    .question-wrap { flex: 1; position: relative; }
    textarea.question-input { width: 100%; padding: 10px 14px; border: 1px solid var(--border); border-radius: var(--radius-sm); font: inherit; font-size: 13px; color: var(--text); background: var(--surface); resize: vertical; min-height: 76px; outline: none; transition: border-color 0.15s; line-height: 1.5; }
    textarea.question-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
    .question-input::placeholder { color: var(--text3); }
    .btn-group { display: flex; flex-direction: column; gap: 8px; }
    .btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 9px 16px; border-radius: var(--radius-sm); font: inherit; font-size: 13px; font-weight: 500; cursor: pointer; border: 1px solid transparent; transition: all 0.12s; white-space: nowrap; }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .btn-primary { background: var(--accent); color: white; border-color: var(--accent); }
    .btn-primary:hover:not(:disabled) { background: #4338ca; }
    .btn-outline { background: var(--surface); color: var(--text); border-color: var(--border2); }
    .btn-outline:hover:not(:disabled) { background: var(--surface2); border-color: var(--border2); }
    .btn svg { width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

    /* ── Output ── */
    .output-area { min-height: 120px; font-size: 13px; line-height: 1.7; color: var(--text); white-space: pre-wrap; word-break: break-word; }
    .output-area.empty { color: var(--text3); font-style: italic; }
    .output-area.loading { color: var(--text2); }

    /* ── Charts ── */
    .charts-grid { display: flex; flex-direction: column; gap: 16px; }
    .chart-wrap { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }

    /* ── Report link ── */
    .report-link { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--accent-text); text-decoration: none; padding: 6px 12px; border: 1px solid var(--accent); border-radius: var(--radius-sm); background: var(--accent-light); margin-top: 12px; transition: background 0.12s; }
    .report-link:hover { background: #e0e7ff; }
    .report-link svg { width: 13px; height: 13px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

    /* ── Tag ── */
    .tag { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500; }
    .tag-indigo { background: var(--accent-light); color: var(--accent-text); }

    /* ── Tabs ── */
    .tabs { display: flex; gap: 2px; padding: 12px 18px 0; border-bottom: 1px solid var(--border); }
    .tab { padding: 8px 14px; font-size: 13px; color: var(--text2); cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; transition: color 0.12s; }
    .tab.active { color: var(--accent-text); border-color: var(--accent); font-weight: 500; }
    .tab:hover:not(.active) { color: var(--text); }
    .tab-content { display: none; padding: 18px; }
    .tab-content.active { display: block; }

    /* ── Suggestion chips ── */
    .suggestions { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .suggestion-chip { background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 5px 12px; font-size: 12px; color: var(--text2); cursor: pointer; transition: all 0.12s; white-space: nowrap; }
    .suggestion-chip:hover { background: var(--accent-light); border-color: var(--accent); color: var(--accent-text); }

    /* ── Data table ── */
    .table-wrap { overflow-x: auto; border-radius: var(--radius-sm); border: 1px solid var(--border); }
    .data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .data-table th { background: var(--surface2); padding: 8px 12px; text-align: left; font-weight: 600; color: var(--text2); border-bottom: 1px solid var(--border); white-space: nowrap; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    .data-table td { padding: 7px 12px; border-bottom: 1px solid var(--border); color: var(--text); white-space: nowrap; }
    .data-table tr:last-child td { border-bottom: none; }
    .data-table tr:hover td { background: var(--surface2); }
    .table-footer { padding: 8px 12px; font-size: 11px; color: var(--text3); background: var(--surface2); border-top: 1px solid var(--border); }

    @media (max-width: 900px) { .layout { flex-direction: column; } .sidebar { width: 100%; flex-direction: row; flex-wrap: wrap; padding: 8px 12px; } .stats-bar { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>

<!-- Topbar -->
<header class="topbar">
  <div class="topbar-logo">
    <div class="topbar-icon">
      <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    </div>
    <span class="topbar-name">Data Analysis Agent</span>
    <span class="topbar-badge">beta</span>
  </div>
  <div class="topbar-status">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">Chờ file...</span>
  </div>
</header>

<div class="layout">

  <!-- Sidebar -->
  <nav class="sidebar">
    <span class="sidebar-section">Workspace</span>
    <button class="nav-item" id="nav-preview" onclick="switchTab('preview')">
      <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
      Data
    </button>
    <button class="nav-item active" id="nav-analyze" onclick="switchTab('analyze')">
      <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
      Analyze
    </button>
    <button class="nav-item" id="nav-chat" onclick="switchTab('chat')">
      <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      Ask Dataset
    </button>
    <button class="nav-item" id="nav-report" onclick="switchTab('report')">
      <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      Report
    </button>

    <div class="sidebar-divider"></div>
    <span class="sidebar-section">Dataset</span>
    <div class="sidebar-file" id="sidebar-file-info" style="display:none">
      <div class="sidebar-file-name" id="sidebar-filename">—</div>
      <div class="sidebar-file-meta" id="sidebar-filemeta">—</div>
    </div>
    <div style="padding: 4px 10px; font-size: 12px; color: var(--text3);" id="sidebar-no-file">Chưa có file</div>
  </nav>

  <!-- Main -->
  <main class="main" id="main-content">

    <!-- Upload panel -->
    <div class="panel" id="upload-panel">
      <div class="panel-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span class="panel-title">Upload Dataset</span>
      </div>
      <div class="panel-body">
        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <div class="upload-zone-icon">
            <svg viewBox="0 0 24 24"><path d="M4 16l4-4 4 4"/><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"/><path d="M12 12V4"/><path d="M8 8l4-4 4 4"/></svg>
          </div>
          <h3>Kéo thả file vào đây</h3>
          <p>Hỗ trợ CSV, XLSX, XLS · tối đa 10MB</p>
          <button class="upload-zone-btn" onclick="event.stopPropagation(); document.getElementById('file-input').click()">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Chọn file
          </button>
          <input type="file" id="file-input" accept=".csv,.xlsx,.xls" multiple />
        </div>
      </div>
    </div>

    <!-- Stats bar (hidden until upload) -->
    <div class="stats-bar" id="stats-bar" style="display:none">
      <div class="stat-card">
        <div class="stat-label">Rows</div>
        <div class="stat-value" id="stat-rows">—</div>
        <div class="stat-sub">records</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Columns</div>
        <div class="stat-value" id="stat-cols">—</div>
        <div class="stat-sub">features</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Numeric</div>
        <div class="stat-value" id="stat-numeric">—</div>
        <div class="stat-sub">columns</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Missing</div>
        <div class="stat-value" id="stat-missing">—</div>
        <div class="stat-sub">null cells</div>
      </div>
    </div>

    <!-- Data Preview panel -->
    <div class="panel" id="tab-preview" style="display:none">
      <div class="panel-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
        <span class="panel-title">Data Preview</span>
        <span id="preview-badge" class="tag tag-indigo" style="margin-left:auto"></span>
      </div>
      <div class="panel-body" style="padding: 0;">
        <div class="table-wrap" id="preview-table-wrap"></div>
      </div>
    </div>

    <!-- Analyze tab -->
    <div class="panel" id="tab-analyze" style="display:none">
      <div class="panel-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        <span class="panel-title">Auto Analyze</span>
        <span class="tag tag-indigo" style="margin-left:auto">Groq AI</span>
      </div>
      <div class="panel-body">
        <div id="suggestions-wrap" style="display:none; margin-bottom:12px;">
          <div style="font-size:11px; color:var(--text3); font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">Gợi ý câu hỏi</div>
          <div class="suggestions" id="suggestion-chips"></div>
        </div>
        <div class="question-row">
          <div class="question-wrap">
            <textarea class="question-input" id="analyze-question" placeholder="Ví dụ: tổng amount theo category, top 10 employee chi tiêu nhiều nhất, trend theo tháng..."></textarea>
          </div>
          <div class="btn-group">
            <button class="btn btn-primary" id="btn-analyze" disabled onclick="doAnalyze()">
              <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              Analyze
            </button>
            <button class="btn btn-outline" id="btn-auto" disabled onclick="doAutoAnalyze()">
              Auto
            </button>
          </div>
        </div>
        <div id="analyze-output-wrap" style="display:none; margin-top:18px; border-top: 1px solid var(--border); padding-top:16px;">
          <div class="output-area" id="analyze-output"></div>
          <div id="analyze-report"></div>
        </div>
        <div id="analyze-charts" class="charts-grid" style="margin-top:16px;"></div>
      </div>
    </div>

    <!-- Chat tab -->
    <div class="panel" id="tab-chat" style="display:none">
      <div class="panel-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="panel-title">Ask Dataset</span>
        <span class="tag tag-indigo" style="margin-left:auto">Groq AI</span>
      </div>
      <div style="padding: 0 18px 8px; max-height: 380px; overflow-y: auto;" id="chat-messages">
        <div style="padding: 24px 0; text-align:center; color: var(--text3); font-size:13px;">Hỏi gì đó về dataset của bạn...</div>
      </div>
      <div style="padding: 12px 18px; border-top: 1px solid var(--border); display: flex; gap: 8px;">
        <textarea class="question-input" id="chat-question" placeholder="Nhập câu hỏi..." style="min-height:44px; max-height:120px;"></textarea>
        <button class="btn btn-primary" id="btn-chat" disabled onclick="doChat()" style="align-self:flex-end; height:44px; padding: 0 16px;">
          <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
    </div>

    <!-- Report tab -->
    <div class="panel" id="tab-report" style="display:none">
      <div class="panel-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span class="panel-title">Markdown Report</span>
      </div>
      <div class="panel-body">
        <div id="report-placeholder" style="color: var(--text3); font-size:13px;">Chạy Analyze trước để tạo report.</div>
        <div id="report-download" style="display:none"></div>
      </div>
    </div>

  </main>
</div>

<script>
  let sessionId = null;
  let currentTab = 'analyze';

  // ── Drag & drop ──
  const dropZone = document.getElementById('drop-zone');
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    if (files.length) uploadFiles(files);
  });
  document.getElementById('file-input').onchange = e => {
    const files = Array.from(e.target.files);
    if (files.length) uploadFiles(files);
  };

  // ── Upload ──
  async function uploadFiles(files) {
    setStatus('uploading');
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) { setStatus('error'); alert(data.detail || 'Upload lỗi.'); return; }
      sessionId = data.session_id;
      const p = data.profile;
      document.getElementById('stat-rows').textContent = (p.rows || 0).toLocaleString();
      document.getElementById('stat-cols').textContent = p.columns || 0;
      document.getElementById('stat-numeric').textContent = p.numeric_columns || 0;
      document.getElementById('stat-missing').textContent = (p.missing_cells || 0).toLocaleString();
      document.getElementById('stats-bar').style.display = 'grid';
      document.getElementById('tab-analyze').style.display = 'none';
      document.getElementById('tab-chat').style.display = 'none';
      document.getElementById('tab-report').style.display = 'none';
      document.getElementById('btn-analyze').disabled = false;
      document.getElementById('btn-auto').disabled = false;
      document.getElementById('btn-chat').disabled = false;
      const fname = files.map(f => f.name).join(', ');
      document.getElementById('sidebar-filename').textContent = fname;
      document.getElementById('sidebar-filemeta').textContent = `${(p.rows||0).toLocaleString()} rows · ${p.columns||0} cols`;
      document.getElementById('sidebar-file-info').style.display = 'block';
      document.getElementById('sidebar-no-file').style.display = 'none';
      if (data.preview_columns && data.preview_rows) renderPreviewTable(data.preview_columns, data.preview_rows, p.rows);
      if (data.suggested_queries && data.suggested_queries.length) renderSuggestions(data.suggested_queries);
      setStatus('ready');
      switchTab('preview');
    } catch(err) { setStatus('error'); alert('Lỗi kết nối: ' + err.message); }
  }

  // ── Analyze ──
  async function doAnalyze() {
    if (!sessionId) return;
    const q = document.getElementById('analyze-question').value.trim() || null;
    setOutputLoading('analyze', 'Đang phân tích...');
    try {
      const res = await fetch('/api/analyze', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId, question: q}) });
      const data = await res.json();
      if (!res.ok) { setOutputLoading('analyze', ''); document.getElementById('analyze-output').textContent = data.detail || 'Lỗi analyze.'; return; }
      document.getElementById('analyze-output-wrap').style.display = 'block';
      document.getElementById('analyze-output').textContent = data.answer;
      document.getElementById('analyze-output').className = 'output-area';
      if (data.report_id) {
        document.getElementById('analyze-report').innerHTML = '<a class="report-link" href="/api/report/' + data.report_id + '" target="_blank"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>Download Markdown report</a>';
        document.getElementById('report-placeholder').style.display = 'none';
        document.getElementById('report-download').style.display = 'block';
        document.getElementById('report-download').innerHTML = '<p style="color:var(--text2);font-size:13px;margin-bottom:12px;">Report đã sẵn sàng sau lần analyze gần nhất.</p><a class="report-link" href="/api/report/' + data.report_id + '" target="_blank"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>Download report.md</a>';
      }
      renderCharts(data.charts || [], 'analyze-charts');
    } catch(err) { document.getElementById('analyze-output').textContent = 'Lỗi: ' + err.message; }
  }

  async function doAutoAnalyze() {
    document.getElementById('analyze-question').value = '';
    await doAnalyze();
  }

  // ── Chat ──
  async function doChat() {
    if (!sessionId) return;
    const q = document.getElementById('chat-question').value.trim();
    if (!q) return;
    addChatMsg(q, 'user');
    document.getElementById('chat-question').value = '';
    addChatMsg('Đang xử lý...', 'bot', 'pending');
    try {
      const res = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId, question: q}) });
      const data = await res.json();
      removePending();
      if (!res.ok) { addChatMsg(data.detail || 'Lỗi.', 'bot'); return; }
      addChatMsg(data.answer, 'bot');
    } catch(err) { removePending(); addChatMsg('Lỗi: ' + err.message, 'bot'); }
  }

  document.getElementById('chat-question').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doChat(); } });

  function addChatMsg(text, role, id) {
    const box = document.getElementById('chat-messages');
    const firstPlaceholder = box.querySelector('div[style*="text-align:center"]');
    if (firstPlaceholder) firstPlaceholder.remove();
    const el = document.createElement('div');
    if (id) el.id = id;
    el.style.cssText = role === 'user'
      ? 'margin: 8px 0 8px auto; max-width:80%; background:var(--accent-light); color:var(--accent-text); padding:10px 14px; border-radius:12px 12px 2px 12px; font-size:13px; line-height:1.5;'
      : 'margin: 8px auto 8px 0; max-width:85%; background:var(--surface2); color:var(--text); padding:10px 14px; border-radius:12px 12px 12px 2px; font-size:13px; line-height:1.5; white-space:pre-wrap;';
    el.textContent = text;
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
  }

  function removePending() {
    const el = document.getElementById('pending');
    if (el) el.remove();
  }

  // ── Charts ──
  function renderCharts(charts, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!charts.length || !window.Plotly) return;
    charts.forEach(chart => {
      const wrap = document.createElement('div');
      wrap.className = 'chart-wrap';
      container.appendChild(wrap);
      const layout = Object.assign({}, chart.plotly_json.layout, { margin: {l:48,r:24,t:40,b:60}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', font: {family:'Inter,sans-serif',size:12,color:'#6b7280'} });
      Plotly.newPlot(wrap, chart.plotly_json.data, layout, { responsive: true, displayModeBar: false });
    });
  }

  // ── Suggestions ──
  function renderSuggestions(queries) {
    const wrap = document.getElementById('suggestions-wrap');
    const chips = document.getElementById('suggestion-chips');
    chips.innerHTML = '';
    queries.forEach(q => {
      const chip = document.createElement('button');
      chip.className = 'suggestion-chip';
      chip.textContent = q;
      chip.onclick = () => {
        document.getElementById('analyze-question').value = q;
        switchTab('analyze');
        document.getElementById('analyze-question').focus();
      };
      chips.appendChild(chip);
    });
    wrap.style.display = 'block';
  }

  // ── Preview table ──
  function renderPreviewTable(columns, rows, totalRows) {
    const badge = document.getElementById('preview-badge');
    badge.textContent = `${totalRows.toLocaleString()} rows`;
    const wrap = document.getElementById('preview-table-wrap');
    let html = '<table class="data-table"><thead><tr>';
    columns.forEach(c => { html += `<th>${escHtml(c)}</th>`; });
    html += '</tr></thead><tbody>';
    rows.forEach(row => {
      html += '<tr>';
      columns.forEach(c => { html += `<td>${escHtml(row[c] ?? '')}</td>`; });
      html += '</tr>';
    });
    html += '</tbody></table>';
    if (totalRows > rows.length) html += `<div class="table-footer">Hiển thị ${rows.length} / ${totalRows.toLocaleString()} dòng</div>`;
    wrap.innerHTML = html;
    document.getElementById('tab-preview').style.display = 'block';
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ── Tabs ──
  function switchTab(tab) {
    currentTab = tab;
    ['preview','analyze','chat','report'].forEach(t => {
      const el = document.getElementById('nav-' + t);
      if (el) el.classList.toggle('active', t === tab);
    });
    if (!sessionId) return;
    document.getElementById('tab-preview').style.display = tab === 'preview' ? 'block' : 'none';
    document.getElementById('tab-analyze').style.display = tab === 'analyze' ? 'block' : 'none';
    document.getElementById('tab-chat').style.display = tab === 'chat' ? 'block' : 'none';
    document.getElementById('tab-report').style.display = tab === 'report' ? 'block' : 'none';
  }

  // ── Status ──
  function setStatus(state) {
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (state === 'ready') { dot.className = 'status-dot ready'; txt.textContent = 'Sẵn sàng'; }
    else if (state === 'uploading') { dot.className = 'status-dot'; dot.style.background='#f59e0b'; txt.textContent = 'Đang upload...'; }
    else if (state === 'error') { dot.className = 'status-dot'; dot.style.background='var(--danger)'; txt.textContent = 'Lỗi'; }
  }

  function setOutputLoading(type, msg) {
    const el = document.getElementById(type + '-output');
    if (el) { el.textContent = msg; el.className = 'output-area loading'; }
    document.getElementById(type + '-output-wrap').style.display = msg ? 'block' : 'none';
  }
</script>
</body>
</html>"""
