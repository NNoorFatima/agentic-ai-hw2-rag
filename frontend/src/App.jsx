import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000/api";

const PIPELINES = [
  {
    id: "rag_fusion",
    label: "RAG Fusion",
    color: "#6366f1",
    desc: "Multi-query fusion with Reciprocal Rank Fusion (RRF)",
  },
  {
    id: "hyde",
    label: "HyDE",
    color: "#10b981",
    desc: "Hypothetical Document Embedding",
  },
  {
    id: "crag",
    label: "CRAG",
    color: "#f59e0b",
    desc: "Corrective RAG with confidence-based routing",
  },
  {
    id: "graph_rag",
    label: "Graph RAG",
    color: "#ef4444",
    desc: "Graph-augmented retrieval",
  },
];

// ── Styles ───────────────────────────────────────────────────────────────────
const styles = {
  app: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #0f1117 0%, #161b27 100%)",
    padding: "0",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  header: {
    background: "rgba(255,255,255,0.03)",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
    padding: "18px 32px",
    display: "flex",
    alignItems: "center",
    gap: "14px",
  },
  headerTitle: {
    fontSize: "22px",
    fontWeight: 700,
    color: "#e1e4e8",
    letterSpacing: "-0.3px",
  },
  headerSub: { fontSize: "13px", color: "#8b949e", marginTop: "2px" },
  badge: {
    background: "#6366f1",
    color: "#fff",
    borderRadius: "6px",
    padding: "3px 9px",
    fontSize: "12px",
    fontWeight: 600,
  },
  main: { maxWidth: "1100px", margin: "0 auto", padding: "32px 24px" },
  card: {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "14px",
    padding: "24px",
    marginBottom: "20px",
  },
  label: {
    fontSize: "12px",
    fontWeight: 600,
    color: "#8b949e",
    textTransform: "uppercase",
    letterSpacing: "0.6px",
    marginBottom: "10px",
  },
  textarea: {
    width: "100%",
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "10px",
    color: "#e1e4e8",
    fontSize: "15px",
    padding: "14px 16px",
    resize: "vertical",
    minHeight: "70px",
    outline: "none",
    fontFamily: "inherit",
    lineHeight: 1.5,
  },
  pipelineGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
    gap: "10px",
    marginTop: "4px",
  },
  pipelineBtn: (selected, color) => ({
    background: selected ? `${color}22` : "rgba(255,255,255,0.04)",
    border: `1.5px solid ${selected ? color : "rgba(255,255,255,0.1)"}`,
    borderRadius: "10px",
    padding: "12px 14px",
    cursor: "pointer",
    textAlign: "left",
    transition: "all 0.15s",
    color: "#e1e4e8",
  }),
  pipelineName: (selected, color) => ({
    fontWeight: 700,
    fontSize: "14px",
    color: selected ? color : "#e1e4e8",
  }),
  pipelineDesc: { fontSize: "12px", color: "#8b949e", marginTop: "3px" },
  runBtn: (loading) => ({
    background: loading ? "#374151" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
    color: "#fff",
    border: "none",
    borderRadius: "10px",
    padding: "13px 32px",
    fontSize: "15px",
    fontWeight: 600,
    cursor: loading ? "not-allowed" : "pointer",
    width: "100%",
    marginTop: "4px",
    transition: "all 0.15s",
    letterSpacing: "0.2px",
  }),
  samplesRow: {
    display: "flex",
    gap: "8px",
    flexWrap: "wrap",
    marginTop: "10px",
  },
  sampleChip: {
    background: "rgba(99,102,241,0.12)",
    border: "1px solid rgba(99,102,241,0.25)",
    borderRadius: "20px",
    padding: "5px 13px",
    fontSize: "12px",
    color: "#a5b4fc",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  sectionTitle: {
    fontSize: "16px",
    fontWeight: 700,
    color: "#e1e4e8",
    marginBottom: "16px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  answerBox: {
    background: "rgba(16,185,129,0.06)",
    border: "1px solid rgba(16,185,129,0.2)",
    borderRadius: "10px",
    padding: "18px 20px",
    fontSize: "15px",
    lineHeight: 1.7,
    color: "#d1fae5",
    whiteSpace: "pre-wrap",
  },
  metaRow: {
    display: "flex",
    gap: "16px",
    flexWrap: "wrap",
    marginBottom: "16px",
  },
  metaBadge: (color) => ({
    background: `${color}22`,
    border: `1px solid ${color}44`,
    borderRadius: "6px",
    padding: "4px 10px",
    fontSize: "12px",
    color: color,
    fontWeight: 600,
  }),
  chunkCard: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: "10px",
    padding: "14px 16px",
    marginBottom: "10px",
  },
  chunkHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "8px",
  },
  chunkSource: {
    fontSize: "12px",
    color: "#8b949e",
    maxWidth: "70%",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  scoreBar: (score) => ({
    height: "5px",
    background: `linear-gradient(90deg, #6366f1 ${Math.round(score * 100)}%, rgba(255,255,255,0.08) 0%)`,
    borderRadius: "3px",
    marginBottom: "8px",
  }),
  chunkText: {
    fontSize: "13px",
    color: "#c9d1d9",
    lineHeight: 1.65,
  },
  variantChip: {
    display: "inline-block",
    background: "rgba(99,102,241,0.1)",
    border: "1px solid rgba(99,102,241,0.2)",
    borderRadius: "6px",
    padding: "4px 10px",
    fontSize: "12px",
    color: "#a5b4fc",
    marginRight: "8px",
    marginBottom: "6px",
  },
  errorBox: {
    background: "rgba(239,68,68,0.08)",
    border: "1px solid rgba(239,68,68,0.25)",
    borderRadius: "10px",
    padding: "14px 16px",
    color: "#fca5a5",
    fontSize: "14px",
  },
  spinner: {
    display: "inline-block",
    width: "18px",
    height: "18px",
    border: "2px solid rgba(255,255,255,0.2)",
    borderTopColor: "#fff",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
    verticalAlign: "middle",
    marginRight: "8px",
  },
};

// ── Utility ───────────────────────────────────────────────────────────────────
function ScoreBar({ score }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  const color = score >= 0.7 ? "#10b981" : score >= 0.4 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div
        style={{
          flex: 1,
          height: "5px",
          background: `linear-gradient(90deg, ${color} ${pct}%, rgba(255,255,255,0.08) 0%)`,
          borderRadius: "3px",
        }}
      />
      <span style={{ fontSize: "11px", color: "#8b949e", minWidth: "35px", textAlign: "right" }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

function ChunkList({ chunks }) {
  const [expanded, setExpanded] = useState({});
  if (!chunks || chunks.length === 0)
    return <p style={{ color: "#8b949e", fontSize: "14px" }}>No chunks retrieved.</p>;

  return chunks.map((c, i) => (
    <div key={i} style={styles.chunkCard}>
      <div style={styles.chunkHeader}>
        <span style={{ fontSize: "12px", fontWeight: 700, color: "#8b949e" }}>
          Chunk #{i + 1}
        </span>
        <a
          href={c.source_url}
          target="_blank"
          rel="noreferrer"
          style={{ fontSize: "12px", color: "#6366f1", maxWidth: "65%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}
          title={c.source_url}
        >
          {c.source_name || c.source_url || "Unknown source"}
        </a>
      </div>
      <ScoreBar score={c.score} />
      <p style={styles.chunkText}>
        {expanded[i] ? c.text : c.text.slice(0, 200) + (c.text.length > 200 ? "…" : "")}
      </p>
      {c.text.length > 200 && (
        <button
          onClick={() => setExpanded((e) => ({ ...e, [i]: !e[i] }))}
          style={{ background: "none", border: "none", color: "#6366f1", fontSize: "12px", cursor: "pointer", marginTop: "6px" }}
        >
          {expanded[i] ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  ));
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState("");
  const [selectedPipeline, setSelectedPipeline] = useState("rag_fusion");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/samples`)
      .then((r) => r.json())
      .then((d) => setSamples(d.samples || []))
      .catch(() => {});
  }, []);

  const handleRun = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim(), pipeline: selectedPipeline, top_k: 5 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Unknown error");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [query, selectedPipeline]);

  const activePipeline = PIPELINES.find((p) => p.id === selectedPipeline);

  return (
    <div style={styles.app}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Header */}
      <div style={styles.header}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={styles.headerTitle}>RAG Pipeline Explorer</span>
            <span style={styles.badge}>CS-4015</span>
          </div>
          <div style={styles.headerSub}>
            Compare RAG Fusion · HyDE · CRAG · Graph RAG on the CRAG corpus
          </div>
        </div>
      </div>

      <div style={styles.main}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
          {/* Left: Input Panel */}
          <div>
            <div style={styles.card}>
              <div style={styles.label}>Your Question</div>
              <textarea
                style={styles.textarea}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. Who directed Inception?"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleRun();
                  }
                }}
              />
              {samples.length > 0 && (
                <>
                  <div style={{ ...styles.label, marginTop: "14px" }}>Sample Queries</div>
                  <div style={styles.samplesRow}>
                    {samples.slice(0, 8).map((s, i) => (
                      <span
                        key={i}
                        style={styles.sampleChip}
                        onClick={() => setQuery(s.query)}
                      >
                        {s.query.length > 50 ? s.query.slice(0, 50) + "…" : s.query}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div style={styles.card}>
              <div style={styles.label}>Select Pipeline</div>
              <div style={styles.pipelineGrid}>
                {PIPELINES.map((p) => (
                  <button
                    key={p.id}
                    style={styles.pipelineBtn(selectedPipeline === p.id, p.color)}
                    onClick={() => setSelectedPipeline(p.id)}
                  >
                    <div style={styles.pipelineName(selectedPipeline === p.id, p.color)}>
                      {p.label}
                    </div>
                    <div style={styles.pipelineDesc}>{p.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <button style={styles.runBtn(loading)} onClick={handleRun} disabled={loading}>
              {loading ? (
                <>
                  <span style={styles.spinner} />
                  Running {activePipeline?.label}…
                </>
              ) : (
                `▶  Run ${activePipeline?.label}`
              )}
            </button>
          </div>

          {/* Right: Results Panel */}
          <div>
            {error && (
              <div style={styles.card}>
                <div style={styles.errorBox}>⚠ {error}</div>
              </div>
            )}

            {result && (
              <>
                {/* Answer */}
                <div style={styles.card}>
                  <div style={styles.sectionTitle}>
                    <span>💬</span> Answer
                    <span
                      style={{
                        ...styles.metaBadge(activePipeline?.color || "#6366f1"),
                        fontSize: "11px",
                      }}
                    >
                      {result.pipeline}
                    </span>
                  </div>

                  {/* Pipeline-specific metadata */}
                  <div style={styles.metaRow}>
                    {result.confidence !== undefined && (
                      <span style={styles.metaBadge("#f59e0b")}>
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                    {result.correction_path && (
                      <span
                        style={styles.metaBadge(
                          result.correction_path === "correct"
                            ? "#10b981"
                            : result.correction_path === "fallback_single"
                            ? "#f59e0b"
                            : "#ef4444"
                        )}
                      >
                        Path: {result.correction_path}
                      </span>
                    )}
                    {result.expanded_count !== undefined && (
                      <span style={styles.metaBadge("#ef4444")}>
                        Graph nodes: {result.expanded_count}
                      </span>
                    )}
                  </div>

                  <div style={styles.answerBox}>{result.answer || "No answer generated."}</div>

                  {/* HyDE: show hypothetical doc */}
                  {result.hypothetical_doc && (
                    <details style={{ marginTop: "12px" }}>
                      <summary
                        style={{ cursor: "pointer", color: "#10b981", fontSize: "13px", fontWeight: 600 }}
                      >
                        Hypothetical Document (used for retrieval)
                      </summary>
                      <div
                        style={{
                          marginTop: "8px",
                          padding: "12px",
                          background: "rgba(16,185,129,0.05)",
                          borderRadius: "8px",
                          fontSize: "13px",
                          color: "#6ee7b7",
                          lineHeight: 1.65,
                        }}
                      >
                        {result.hypothetical_doc}
                      </div>
                    </details>
                  )}

                  {/* RAG Fusion: show query variants */}
                  {result.query_variants && result.query_variants.length > 0 && (
                    <details style={{ marginTop: "12px" }}>
                      <summary
                        style={{ cursor: "pointer", color: "#6366f1", fontSize: "13px", fontWeight: 600 }}
                      >
                        Query Variants ({result.query_variants.length})
                      </summary>
                      <div style={{ marginTop: "8px" }}>
                        {result.query_variants.map((v, i) => (
                          <span key={i} style={styles.variantChip}>
                            {v}
                          </span>
                        ))}
                      </div>
                    </details>
                  )}
                </div>

                {/* Retrieved Chunks */}
                <div style={styles.card}>
                  <div style={styles.sectionTitle}>
                    <span>📄</span> Retrieved Chunks ({result.retrieved?.length || 0})
                  </div>
                  <ChunkList chunks={result.retrieved} />
                </div>
              </>
            )}

            {!result && !error && !loading && (
              <div style={{ ...styles.card, textAlign: "center", padding: "60px 24px" }}>
                <div style={{ fontSize: "40px", marginBottom: "14px" }}>🔍</div>
                <div style={{ color: "#8b949e", fontSize: "15px" }}>
                  Enter a question and select a pipeline to see results
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
