def render_home() -> str:
    return """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data Analysis AI Agent</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #f7f8fa; color: #17202a; }
    header { padding: 20px 28px; background: #111827; color: white; }
    header h1 { margin: 0; font-size: 20px; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; display: grid; gap: 18px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 18px; align-items: start; }
    input, textarea, button { font: inherit; }
    input[type=file], textarea { width: 100%; border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; }
    textarea { min-height: 84px; resize: vertical; }
    button { border: 0; border-radius: 8px; background: #2563eb; color: white; padding: 10px 14px; cursor: pointer; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    pre { white-space: pre-wrap; background: #f3f4f6; border-radius: 8px; padding: 12px; overflow: auto; }
    .charts { display: grid; gap: 16px; }
    .chart { min-height: 360px; border: 1px solid #e5e7eb; border-radius: 8px; }
    .meta { color: #6b7280; font-size: 13px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Data Analysis AI Agent</h1>
    <div class="meta">Upload CSV/XLSX → profile → charts → insights → Markdown report</div>
  </header>
  <main class="grid">
    <section>
      <h2>Dataset</h2>
      <input id="file" type="file" accept=".csv,.xlsx,.xls" multiple />
      <p><button id="upload">Upload & Profile</button></p>
      <div id="session" class="meta"></div>
      <h3>Question</h3>
      <textarea id="question" placeholder="Ví dụ: top category phổ biến nhất, tổng sales, trung bình age..."></textarea>
      <p>
        <button id="analyze" disabled>Auto Analyze</button>
        <button id="chat" disabled>Ask Dataset</button>
      </p>
      <div id="report"></div>
    </section>
    <section>
      <h2>Output</h2>
      <pre id="answer">Chưa có dữ liệu.</pre>
      <div id="charts" class="charts"></div>
    </section>
  </main>
  <script>
    let sessionId = null;
    const $ = (id) => document.getElementById(id);
    const setAnswer = (text) => $('answer').textContent = text;

    $('upload').onclick = async () => {
      const files = Array.from($('file').files);
      if (!files.length) return alert('Chọn file CSV/XLSX trước.');
      const form = new FormData();
      files.forEach((file) => form.append('files', file));
      $('upload').disabled = true;
      setAnswer('Đang upload và profiling...');
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      $('upload').disabled = false;
      if (!res.ok) return setAnswer(data.detail || 'Upload lỗi.');
      sessionId = data.session_id;
      $('session').textContent = `Session: ${sessionId} · ${data.profile.rows} rows · ${data.profile.columns} columns`;
      $('analyze').disabled = false;
      $('chat').disabled = false;
      setAnswer(JSON.stringify(data.profile, null, 2));
    };

    $('analyze').onclick = async () => {
      if (!sessionId) return;
      setAnswer('Đang phân tích...');
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, question: $('question').value || null })
      });
      const data = await res.json();
      if (!res.ok) return setAnswer(data.detail || 'Analyze lỗi.');
      setAnswer(data.answer);
      $('report').innerHTML = `<a href="/api/report/${data.report_id}" target="_blank">Download Markdown report</a>`;
      renderCharts(data.charts || []);
    };

    $('chat').onclick = async () => {
      if (!sessionId) return;
      const question = $('question').value.trim();
      if (!question) return alert('Nhập câu hỏi trước.');
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, question })
      });
      const data = await res.json();
      if (!res.ok) return setAnswer(data.detail || 'Chat lỗi.');
      setAnswer(`${data.answer}\\n\\nSQL:\\n${(data.executed_queries || []).join('\\n')}`);
    };

    function renderCharts(charts) {
      $('charts').innerHTML = '';
      if (!window.Plotly) {
        return;
      }
      charts.forEach((chart) => {
        const el = document.createElement('div');
        el.className = 'chart';
        $('charts').appendChild(el);
        Plotly.newPlot(el, chart.plotly_json.data, chart.plotly_json.layout, { responsive: true });
      });
    }
  </script>
</body>
</html>"""
