import { useState, useRef } from "react";

const API_BASE = "/api";

// ============================================================
// UI COMPONENTS
// ============================================================

const AgentCard = ({ result, index }) => {
  const [expanded, setExpanded] = useState(false);
  if (!result) return null;
  const statusColors = { success: "#10b981", error: "#ef4444", pending: "#f59e0b" };
  return (
    <div style={{
      background: "rgba(15,23,42,0.7)", border: `1px solid rgba(148,163,184,0.15)`,
      borderRadius: 12, padding: 20, marginBottom: 12,
      animation: `fadeSlideIn 0.5s ease ${index * 0.08}s both`
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: statusColors[result.status] || "#64748b", boxShadow: `0 0 8px ${statusColors[result.status]}` }} />
          <span style={{ color: "#e2e8f0", fontFamily: "mono", fontSize: 14, fontWeight: 600 }}>{result.agent}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ color: "#94a3b8", fontSize: 12, fontFamily: "mono" }}>{result.processing_time}</span>
          <span style={{ background: `${statusColors[result.status]}22`, color: statusColors[result.status], padding: "2px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, fontFamily: "mono" }}>
            {Math.round((result.confidence || 0) * 100)}% conf
          </span>
        </div>
      </div>
      <div style={{ color: "#64748b", fontSize: 12, marginBottom: 8, fontStyle: "italic" }}>Owner: {result.owner}</div>
      <button onClick={() => setExpanded(!expanded)} style={{
        background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.3)", color: "#06b6d4",
        padding: "6px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12, fontFamily: "mono"
      }}>
        {expanded ? "▼ Hide" : "▶ Show"} Reasoning ({result.reasoning?.length || 0} steps)
      </button>
      {expanded && (
        <div style={{ marginTop: 12, padding: 14, background: "rgba(0,0,0,0.3)", borderRadius: 8, borderLeft: "3px solid #06b6d4" }}>
          {result.reasoning?.map((step, i) => (
            <div key={i} style={{ color: "#94a3b8", fontSize: 12, fontFamily: "mono", padding: "3px 0", lineHeight: 1.6 }}>
              <span style={{ color: "#475569" }}>{String(i + 1).padStart(2, "0")}.</span> {step}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const MetricBox = ({ label, value, color = "#06b6d4", sub = "" }) => (
  <div style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: 12, padding: "18px 20px", flex: "1 1 160px", minWidth: 160 }}>
    <div style={{ color: "#64748b", fontSize: 11, fontFamily: "mono", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{label}</div>
    <div style={{ color, fontSize: 20, fontWeight: 700, fontFamily: "mono", wordBreak: "break-word", lineHeight: 1.2 }}>{value}</div>
    {sub && <div style={{ color: "#475569", fontSize: 11, marginTop: 4 }}>{sub}</div>}
  </div>
);

const ProgressBar = ({ progress, label }) => (
  <div style={{ marginBottom: 6 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
      <span style={{ color: "#94a3b8", fontSize: 12, fontFamily: "mono" }}>{label}</span>
      <span style={{ color: "#06b6d4", fontSize: 12, fontFamily: "mono" }}>{progress}%</span>
    </div>
    <div style={{ background: "rgba(30,41,59,0.8)", borderRadius: 4, height: 6, overflow: "hidden" }}>
      <div style={{ width: `${progress}%`, height: "100%", background: "linear-gradient(90deg,#06b6d4,#22d3ee)", borderRadius: 4, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)" }} />
    </div>
  </div>
);

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [stage, setStage] = useState("upload");
  const [files, setFiles] = useState([]);
  const [progress, setProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState("");
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("pipeline");
  const [dragActive, setDragActive] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const fileInputRef = useRef(null);

  const agentPipeline = [
    { key: "image", name: "Image Processing", icon: "🖼️", owner: "Vivek Vardhan" },
    { key: "pdf", name: "PDF Extraction", icon: "📄", owner: "Swapnil Sontakke" },
    { key: "requirements", name: "Requirements Check", icon: "📋", owner: "Karthikeyan Pillai" },
    { key: "credibility", name: "Credibility & Policy", icon: "🔍", owner: "Shruti Roy" },
    { key: "billing", name: "Billing Analysis", icon: "💰", owner: "Siri Spandana" },
    { key: "fraud", name: "Fraud Detection", icon: "🛡️", owner: "Titash Bhattacharya" },
    { key: "orchestrator", name: "Orchestrator", icon: "🎯", owner: "Aadithya Pabbisetty" },
  ];

  const handleFiles = (newFiles) => setFiles((prev) => [...prev, ...Array.from(newFiles)]);
  const handleDrop = (e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); };

  const resumeClaimWithFiles = async (additionalFiles, claimId) => {
    setStage("processing");
    setProgress(10);
    setProcessingStatus("Uploading additional documents...");

    const formData = new FormData();
    additionalFiles.forEach((f) => formData.append("files", f));
    formData.append("claim_id", claimId);

    const progressInterval = setInterval(() => {
      setProgress((p) => Math.min(p + 2, 85));
    }, 800);

    try {
      const res = await fetch(`${API_BASE}/resume-claim`, { method: "POST", body: formData });
      clearInterval(progressInterval);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Resume failed");
      }

      const data = await res.json();
      
      if (data.status === "still_hold") {
        // Still missing fields - update result with merged data
        setProgress(100);
        setProcessingStatus(`Still missing: ${data.missing_fields.join(", ")}`);
        
        // Update result to show merged extracted fields
        setResult(prev => ({
          ...prev,
          claim_summary: {
            ...prev.claim_summary,
            ...data.extracted_fields,
            status: "PENDING_DOCUMENTS",
            missing_fields: data.missing_fields,
            action_required: "Please upload documents containing the missing information"
          }
        }));
        
        setStage("results"); // Stay on results page
      } else {
        // Requirements met, processing completed
        setResult(data.result);
        setProgress(100);
        setProcessingStatus(""); // Clear status message
        setStage("results");
      }
    } catch (e) {
      clearInterval(progressInterval);
      setProcessingStatus(`Error: ${e.message}`);
      setProgress(0);
    }
  };

  const processClaimPipeline = async () => {
    setStage("processing");
    setProgress(10);
    setProcessingStatus("Uploading files and starting agent pipeline...");

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    // More conservative progress simulation - slower increments
    const progressInterval = setInterval(() => {
      setProgress((p) => Math.min(p + 2, 85)); // Cap at 85% until backend responds
    }, 800);

    const steps = [
      "Image Processing Agent analyzing documents...",
      "PDF Processing Agent extracting text...",
      "Requirements Agent validating documents...",
      "Checking for missing fields...",
    ];
    let stepIdx = 0;
    const statusInterval = setInterval(() => {
      if (stepIdx < steps.length) {
        setProcessingStatus(steps[stepIdx]);
        stepIdx++;
      } else {
        // After initial steps, show waiting message
        setProcessingStatus("Waiting for backend processing...");
      }
    }, 2000);

    try {
      const res = await fetch(`${API_BASE}/process-claim`, { method: "POST", body: formData });
      clearInterval(progressInterval);
      clearInterval(statusInterval);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Processing failed");
      }

      const data = await res.json();
      setResult(data.result);
      setProgress(100);
      setProcessingStatus(""); // Clear the waiting message
      setStage("results");
    } catch (e) {
      clearInterval(progressInterval);
      clearInterval(statusInterval);
      setProcessingStatus(`Error: ${e.message}. Make sure the backend is running on port 8000.`);
      setProgress(0);
    }
  };

  const decisionColors = { APPROVE: "#10b981", REJECT: "#ef4444", REVIEW: "#f59e0b", HOLD: "#8b5cf6" };

  const resetApp = () => {
    setStage("upload"); setFiles([]); setResult(null);
    setProgress(0); setActiveTab("pipeline"); setProcessingStatus("");
  };

  // Extract agent results from the orchestrator reasoning trace
  const getAgentResults = () => {
    if (!result) return [];
    // The backend returns reasoning_trace and agent_summaries
    // We reconstruct individual agent cards from agent_summaries
    return result.agent_summaries || [];
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0a0e1a 0%, #0f172a 40%, #0c1222 100%)",
      color: "#e2e8f0", fontFamily: "'Segoe UI', system-ui, sans-serif",
      position: "relative", overflow: "hidden"
    }}>
      <style>{`
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: rgba(15,23,42,0.5); } ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
      `}</style>

      <div style={{ position: "fixed", top: -200, right: -200, width: 500, height: 500, background: "radial-gradient(circle, rgba(6,182,212,0.06) 0%, transparent 70%)", pointerEvents: "none" }} />

      {/* Header */}
      <header style={{ borderBottom: "1px solid rgba(148,163,184,0.08)", padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", backdropFilter: "blur(20px)", position: "sticky", top: 0, zIndex: 50, background: "rgba(10,14,26,0.85)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: "linear-gradient(135deg, #06b6d4, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>⚕</div>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: -0.5, background: "linear-gradient(90deg, #e2e8f0, #94a3b8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>ClaimFlow AI</h1>
            <p style={{ color: "#475569", fontSize: 11, fontFamily: "mono" }}>Multi-Agent Insurance Claim Processor</p>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {stage !== "upload" && (
            <button onClick={resetApp} style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#ef4444", padding: "8px 16px", borderRadius: 8, cursor: "pointer", fontSize: 13, fontFamily: "mono" }}>↺ New Claim</button>
          )}
        </div>
      </header>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 24px" }}>

        {/* =================== UPLOAD STAGE =================== */}
        {stage === "upload" && (
          <div style={{ animation: "fadeSlideIn 0.5s ease both" }}>
            <div style={{ textAlign: "center", marginBottom: 40 }}>
              <h2 style={{ fontSize: 36, fontWeight: 800, letterSpacing: -1, marginBottom: 12, background: "linear-gradient(90deg, #06b6d4, #8b5cf6, #06b6d4)", backgroundSize: "200% 200%", animation: "gradientShift 4s ease infinite", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                AI-Powered Claim Processing
              </h2>
              <p style={{ color: "#64748b", fontSize: 15, maxWidth: 550, margin: "0 auto", lineHeight: 1.6 }}>
                Upload insurance claim documents to trigger the multi-agent processing pipeline.
                Seven specialized AI agents will analyze, validate, and produce a comprehensive assessment.
              </p>
            </div>

            {/* Pipeline Visual */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flexWrap: "wrap", gap: 6, marginBottom: 40, padding: "0 10px" }}>
              {agentPipeline.map((agent, i) => (
                <div key={agent.key} style={{ display: "flex", alignItems: "center" }}>
                  <div style={{ background: agent.key === "orchestrator" ? "rgba(139,92,246,0.15)" : "rgba(6,182,212,0.08)", border: `1px solid ${agent.key === "orchestrator" ? "rgba(139,92,246,0.4)" : "rgba(6,182,212,0.2)"}`, borderRadius: 10, padding: "10px 14px", textAlign: "center", minWidth: 110 }}>
                    <div style={{ fontSize: 22, marginBottom: 4 }}>{agent.icon}</div>
                    <div style={{ color: "#e2e8f0", fontSize: 11, fontWeight: 600, fontFamily: "mono" }}>{agent.name}</div>
                    <div style={{ color: "#475569", fontSize: 10 }}>{agent.owner}</div>
                  </div>
                  {i < agentPipeline.length - 1 && <span style={{ color: "#1e293b", margin: "0 2px", fontSize: 18 }}>→</span>}
                </div>
              ))}
            </div>

            {/* Upload */}
            <div onDragOver={(e) => { e.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}
              style={{ border: `2px dashed ${dragActive ? "#06b6d4" : "rgba(148,163,184,0.15)"}`, borderRadius: 16, padding: "60px 40px", textAlign: "center", cursor: "pointer", background: dragActive ? "rgba(6,182,212,0.05)" : "rgba(15,23,42,0.4)", transition: "all 0.3s", marginBottom: 20 }}>
              <input ref={fileInputRef} type="file" multiple accept="image/*,.pdf,.doc,.docx" onChange={(e) => handleFiles(e.target.files)} style={{ display: "none" }} />
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.7 }}>📎</div>
              <p style={{ color: "#94a3b8", fontSize: 16, fontWeight: 500, marginBottom: 8 }}>Drop claim documents here or click to browse</p>
              <p style={{ color: "#475569", fontSize: 13 }}>Supports: Medical discharge summaries, policy documents, hospital bills, lab reports</p>
            </div>

            {files.length > 0 && (
              <div style={{ marginBottom: 24 }}>
                <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 10, fontFamily: "mono" }}>{files.length} file(s) selected:</div>
                {files.map((f, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(15,23,42,0.6)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: 8, padding: "10px 16px", marginBottom: 6 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span>{f.type?.includes("image") ? "🖼️" : "📄"}</span>
                      <span style={{ color: "#e2e8f0", fontSize: 13, fontFamily: "mono" }}>{f.name}</span>
                    </div>
                    <span style={{ color: "#475569", fontSize: 12 }}>{(f.size / 1024).toFixed(1)} KB</span>
                  </div>
                ))}
                <button onClick={processClaimPipeline} style={{ width: "100%", marginTop: 16, padding: 16, background: "linear-gradient(135deg, #06b6d4, #0891b2)", border: "none", borderRadius: 12, color: "#fff", fontSize: 16, fontWeight: 700, cursor: "pointer", letterSpacing: 0.5, boxShadow: "0 4px 20px rgba(6,182,212,0.3)" }}>
                  ▶ Process Claim Through Agent Pipeline
                </button>
              </div>
            )}
          </div>
        )}

        {/* =================== PROCESSING STAGE =================== */}
        {stage === "processing" && (
          <div style={{ animation: "fadeSlideIn 0.4s ease both" }}>
            <h2 style={{ fontSize: 22, fontWeight: 700, color: "#e2e8f0", marginBottom: 6 }}>Processing Claim...</h2>
            <ProgressBar progress={progress} label="Pipeline Progress" />
            <div style={{ marginTop: 20, padding: 24, background: "rgba(15,23,42,0.7)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: 12 }}>
              <div style={{ color: "#06b6d4", fontSize: 14, fontFamily: "mono", animation: progress > 0 && progress < 100 ? "pulse 1.5s infinite" : "none" }}>
                ⟳ {processingStatus}
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flexWrap: "wrap", gap: 8, marginTop: 24 }}>
              {agentPipeline.map((agent) => (
                <div key={agent.key} style={{ padding: "8px 14px", borderRadius: 8, fontSize: 12, fontFamily: "mono", background: "rgba(30,41,59,0.5)", border: "1px solid rgba(71,85,105,0.3)", color: "#475569" }}>
                  {agent.icon} {agent.name}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* =================== RESULTS STAGE =================== */}
        {stage === "results" && result && (
          <div style={{ animation: "fadeSlideIn 0.5s ease both" }}>
            {/* Decision Banner */}
            <div style={{ background: `${decisionColors[result.decision]}11`, border: `2px solid ${decisionColors[result.decision]}44`, borderRadius: 16, padding: "28px 32px", marginBottom: 24, textAlign: "center" }}>
              <div style={{ fontSize: 42, fontWeight: 800, color: decisionColors[result.decision], letterSpacing: -1, marginBottom: 8 }}>
                {result.decision === "APPROVE" ? "✅" : result.decision === "REJECT" ? "❌" : result.decision === "REVIEW" ? "⚠️" : "⏸️"} CLAIM {result.decision}
              </div>
              
              {/* Fraud Category Badge (if applicable) */}
              {result.claim_summary?.fraud_category && result.claim_summary.fraud_category !== "NONE" && (
                <div style={{ 
                  display: "inline-block",
                  padding: "6px 16px",
                  borderRadius: 20,
                  fontSize: 13,
                  fontWeight: 700,
                  fontFamily: "mono",
                  marginBottom: 12,
                  background: result.claim_summary.fraud_category === "DUPLICATE_CLAIM" ? "rgba(245,158,11,0.15)" :
                              result.claim_summary.fraud_category === "FRAUD" ? "rgba(239,68,68,0.15)" :
                              "rgba(249,115,22,0.15)",
                  color: result.claim_summary.fraud_category === "DUPLICATE_CLAIM" ? "#f59e0b" :
                         result.claim_summary.fraud_category === "FRAUD" ? "#ef4444" :
                         "#f97316",
                  border: `1px solid ${result.claim_summary.fraud_category === "DUPLICATE_CLAIM" ? "#f59e0b" :
                                       result.claim_summary.fraud_category === "FRAUD" ? "#ef4444" :
                                       "#f97316"}44`
                }}>
                  {result.claim_summary.fraud_category === "DUPLICATE_CLAIM" ? "🔄 Duplicate Claim" :
                   result.claim_summary.fraud_category === "FRAUD" ? "🚨 Fraud Detected" :
                   "⚠️ Suspicious Activity"}
                </div>
              )}
              
              <div style={{ color: "#94a3b8", fontSize: 14, maxWidth: 700, margin: "0 auto", lineHeight: 1.7, marginBottom: result.decision !== "HOLD" ? 16 : 0 }}>
                {result.processing_summary?.ai_summary || result.decision_reasons?.join(". ")}
                
                {/* Show fraud type details if available */}
                {result.claim_summary?.fraud_type_details && result.claim_summary.fraud_type_details.length > 0 && (
                  <div style={{ marginTop: 12, padding: "12px 16px", background: "rgba(239,68,68,0.1)", borderRadius: 8, border: "1px solid rgba(239,68,68,0.2)" }}>
                    <div style={{ color: "#ef4444", fontSize: 12, fontWeight: 600, marginBottom: 6 }}>⚠️ Fraud Indicators:</div>
                    {result.claim_summary.fraud_type_details.map((detail, i) => (
                      <div key={i} style={{ color: "#fca5a5", fontSize: 12, marginLeft: 8 }}>• {detail}</div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Action Buttons - Show for all decisions except HOLD */}
              {result.decision !== "HOLD" && (
                <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
                  {/* Approve Button */}
                  <button 
                    onClick={async () => {
                      try {
                        setProcessingStatus("Storing approved claim in history database...");
                        const response = await fetch(`${API_BASE}/approve-claim`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify(result),
                        });
                        const data = await response.json();
                        if (data.success) {
                          setProcessingStatus(`✅ Claim approved and stored! ID: ${data.claim_id}`);
                          setTimeout(() => setProcessingStatus(""), 3000);
                        } else {
                          setProcessingStatus(`❌ Failed to store claim: ${data.detail || "Unknown error"}`);
                        }
                      } catch (e) {
                        setProcessingStatus(`❌ Error: ${e.message}`);
                      }
                    }}
                    style={{
                      background: "linear-gradient(135deg, #10b981, #059669)",
                      border: "none",
                      color: "#fff",
                      padding: "12px 32px",
                      borderRadius: 10,
                      cursor: "pointer",
                      fontSize: 14,
                      fontWeight: 600,
                      fontFamily: "mono",
                      boxShadow: "0 4px 12px rgba(16,185,129,0.3)",
                      transition: "all 0.2s"
                    }}
                    onMouseOver={(e) => e.target.style.transform = "translateY(-2px)"}
                    onMouseOut={(e) => e.target.style.transform = "translateY(0)"}
                  >
                    ✓ Approve
                  </button>
                  
                  {/* Reject Button */}
                  <button 
                    onClick={async () => {
                      try {
                        setProcessingStatus("Storing rejected claim in history database...");
                        const response = await fetch(`${API_BASE}/reject-claim`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify(result),
                        });
                        const data = await response.json();
                        if (data.success) {
                          setProcessingStatus(`✅ Claim rejected and stored! ID: ${data.claim_id}`);
                          setTimeout(() => setProcessingStatus(""), 3000);
                        } else {
                          setProcessingStatus(`❌ Failed to store claim: ${data.detail || "Unknown error"}`);
                        }
                      } catch (e) {
                        setProcessingStatus(`❌ Error: ${e.message}`);
                      }
                    }}
                    style={{
                      background: "linear-gradient(135deg, #ef4444, #dc2626)",
                      border: "none",
                      color: "#fff",
                      padding: "12px 32px",
                      borderRadius: 10,
                      cursor: "pointer",
                      fontSize: 14,
                      fontWeight: 600,
                      fontFamily: "mono",
                      boxShadow: "0 4px 12px rgba(239,68,68,0.3)",
                      transition: "all 0.2s"
                    }}
                    onMouseOver={(e) => e.target.style.transform = "translateY(-2px)"}
                    onMouseOut={(e) => e.target.style.transform = "translateY(0)"}
                  >
                    ✗ Reject
                  </button>
                </div>
              )}
              
              {/* Status message */}
              {processingStatus && (
                <div style={{ marginTop: 12, color: processingStatus.includes("✅") ? "#10b981" : "#ef4444", fontSize: 13, fontFamily: "mono" }}>
                  {processingStatus}
                </div>
              )}
            </div>

            {/* Metrics */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
              <MetricBox label="Amount Claimed" value={`₹${((result.claim_summary?.amount_claimed || 0) / 1000).toFixed(0)}K`} color="#e2e8f0" sub={`₹${(result.claim_summary?.amount_claimed || 0).toLocaleString("en-IN")}`} />
              <MetricBox label="Amount Approved" value={`₹${((result.claim_summary?.amount_approved || 0) / 1000).toFixed(0)}K`} color="#10b981" sub={`₹${(result.claim_summary?.amount_approved || 0).toLocaleString("en-IN")}`} />
              <MetricBox label="Confidence" value={`${((result.weighted_confidence || 0) * 100).toFixed(1)}%`} color="#06b6d4" sub="Weighted average" />
              <MetricBox label="Fraud Score" value={(result.claim_summary?.fraud_score || 0).toFixed(2)} color={(result.claim_summary?.fraud_score || 0) > 0.3 ? "#ef4444" : "#10b981"} sub={(result.claim_summary?.fraud_score || 0) > 0.3 ? "Elevated" : "Very Low Risk"} />
              <MetricBox
                      label="Billing Items"
                      value={`${result.claim_summary?.billing_summary?.line_items_parsed || 0}`}
                      color="#f59e0b"
                      sub="Rows parsed"
                    />
              <MetricBox 
                label="Fraud Category" 
                value={
                  result.claim_summary?.fraud_category === "DUPLICATE_CLAIM" ? "Duplicate Claim" :
                  result.claim_summary?.fraud_category === "SUSPICIOUS" ? "Suspicious" :
                  result.claim_summary?.fraud_category === "FRAUD" ? "Fraud" :
                  result.claim_summary?.fraud_category === "NONE" ? "None" :
                  result.claim_summary?.fraud_category || "None"
                } 
                color={
                  result.claim_summary?.fraud_category === "DUPLICATE_CLAIM" ? "#f59e0b" :
                  result.claim_summary?.fraud_category === "FRAUD" ? "#ef4444" :
                  result.claim_summary?.fraud_category === "SUSPICIOUS" ? "#f97316" :
                  "#10b981"
                } 
                sub={
                  result.claim_summary?.fraud_category === "DUPLICATE_CLAIM" ? "Duplicate Detected" :
                  result.claim_summary?.fraud_category === "FRAUD" ? "High Risk" :
                  result.claim_summary?.fraud_category === "SUSPICIOUS" ? "Needs Review" :
                  "Clean"
                } 
              />
              <MetricBox label="Processing" value={result.processing_summary?.orchestrator_time || "N/A"} color="#8b5cf6" sub={`${result.processing_summary?.agents_passed}/${result.processing_summary?.total_agents} agents passed`} />
            </div>

            {/* Tabs */}
            <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "rgba(15,23,42,0.5)", padding: 4, borderRadius: 10 }}>
              {[{ key: "pipeline", label: "Decision Fusion" }, { key: "claim", label: "Claim Details" }, { key: "trace", label: "Reasoning Trace" }].map((tab) => (
                <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{ flex: 1, padding: "10px 16px", borderRadius: 8, border: "none", cursor: "pointer", fontSize: 13, fontWeight: 600, fontFamily: "mono", background: activeTab === tab.key ? "rgba(6,182,212,0.15)" : "transparent", color: activeTab === tab.key ? "#06b6d4" : "#64748b" }}>
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Decision Fusion Tab */}
            {activeTab === "pipeline" && (
              <div style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: 12, padding: 24, animation: "fadeSlideIn 0.4s ease both" }}>
                <h3 style={{ color: "#06b6d4", fontSize: 16, fontWeight: 700, marginBottom: 20, fontFamily: "mono" }}>⚖️ Weighted Decision Fusion Matrix</h3>
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr", gap: 1, background: "rgba(148,163,184,0.1)", borderRadius: 8, overflow: "hidden", marginBottom: 24 }}>
                  {["Agent", "Confidence", "Weight", "Weighted", "Status"].map((h) => (
                    <div key={h} style={{ background: "rgba(6,182,212,0.1)", padding: "10px 14px", color: "#06b6d4", fontSize: 11, fontWeight: 700, fontFamily: "mono", textTransform: "uppercase" }}>{h}</div>
                  ))}
                  {(result.agent_summaries || []).map((a, i) => (
                    [a.agent, `${(a.confidence * 100).toFixed(1)}%`, a.weight, a.weighted_score?.toFixed(4) || "0", a.status].map((cell, j) => (
                      <div key={`${i}-${j}`} style={{ background: "rgba(15,23,42,0.9)", padding: "10px 14px", color: j === 4 ? (cell === "PASS" ? "#10b981" : "#ef4444") : "#94a3b8", fontSize: 12, fontFamily: "mono", fontWeight: j === 4 ? 700 : 400 }}>{cell}</div>
                    ))
                  ))}
                </div>
                <h4 style={{ color: "#e2e8f0", fontSize: 14, marginBottom: 14, fontFamily: "mono" }}>Confidence Distribution</h4>
                {(result.agent_summaries || []).map((a, i) => (
                  <ProgressBar key={i} progress={Math.round(a.confidence * 100)} label={a.agent} />
                ))}
                <div style={{ marginTop: 24, padding: 20, background: `${decisionColors[result.decision]}08`, border: `1px solid ${decisionColors[result.decision]}33`, borderRadius: 10, textAlign: "center" }}>
                  <div style={{ color: "#64748b", fontSize: 12, marginBottom: 6, fontFamily: "mono" }}>FINAL WEIGHTED CONFIDENCE</div>
                  <div style={{ fontSize: 40, fontWeight: 800, color: decisionColors[result.decision], fontFamily: "mono" }}>{((result.weighted_confidence || 0) * 100).toFixed(1)}%</div>
                </div>
              </div>
            )}

            {/* Claim Details Tab */}
            {activeTab === "claim" && (
              <div style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: 12, padding: 24, animation: "fadeSlideIn 0.4s ease both" }}>
                <h3 style={{ color: "#e2e8f0", fontSize: 16, fontWeight: 700, marginBottom: 20, fontFamily: "mono" }}>Claim Summary</h3>
                
                {/* Show upload button for HOLD claims */}
                {result.decision === "HOLD" && result.claim_summary?.missing_fields && (
                  <div style={{ marginBottom: 24, padding: 20, background: "rgba(139,92,246,0.1)", border: "2px solid rgba(139,92,246,0.3)", borderRadius: 12 }}>
                    <div style={{ color: "#a78bfa", fontSize: 14, fontWeight: 600, marginBottom: 12 }}>📋 Missing Required Information</div>
                    <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 16 }}>
                      The following fields are missing: <strong style={{ color: "#e2e8f0" }}>{result.claim_summary.missing_fields.join(", ")}</strong>
                    </div>
                    <label style={{ display: "block", width: "100%", padding: "14px 20px", background: "linear-gradient(135deg, #8b5cf6, #7c3aed)", borderRadius: 10, textAlign: "center", color: "#fff", fontSize: 14, fontWeight: 600, cursor: "pointer", transition: "all 0.2s" }}>
                      📤 Upload Additional Documents
                      <input type="file" multiple accept="image/*,application/pdf" onChange={(e) => {
                        const additionalFiles = Array.from(e.target.files);
                        if (additionalFiles.length > 0) {
                          resumeClaimWithFiles(additionalFiles, result.claim_summary?.claim_id || "unknown");
                        }
                      }} style={{ display: "none" }} />
                    </label>
                  </div>
                )}

                {/* Adding Billing details. Modified by Spandana */}
                {/* Billing Details */}
                {result?.claim_summary && (
                <div style={{
                marginTop: 24,
                marginBottom: 24,
                padding: 20,
                background: "rgba(15,23,42,0.55)",
                border: "1px solid rgba(6,182,212,0.2)",
                borderRadius: 12
                }}>
                <h3 style={{
                color: "#e2e8f0",
                fontSize: 16,
                fontWeight: 700,
                marginBottom: 16
                }}>
                Billing Details
                </h3>

                <div style={{ marginBottom: 16 }}>
                <div style={{ color: "#cbd5e1", marginBottom: 6 }}>
                Billing Anomaly Score: {Number(result.claim_summary.billing_anomaly_score || 0).toFixed(2)}
                </div>
                <div style={{ color: "#cbd5e1", marginBottom: 6 }}>
                Billing Deductions: ₹{Number(result.claim_summary.billing_deductions || 0).toLocaleString("en-IN")}
                </div>
                </div>
                
                {result.claim_summary.billing_summary && (
                  <div
                    style={{
                      marginBottom: 16,
                      padding: 14,
                      borderRadius: 10,
                      background: "rgba(30,41,59,0.55)",
                      border: "1px solid rgba(148,163,184,0.15)"
                    }}
                  >
                    <div style={{ color: "#e2e8f0", fontWeight: 700, marginBottom: 10 }}>
                      Billing Summary
                    </div>

                    <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 6 }}>
                      Line Items Parsed: {Number(result.claim_summary.billing_summary.line_items_parsed || 0)}
                    </div>

                    <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 6 }}>
                      Categories Detected: {
                        result.claim_summary.billing_summary.categories_detected?.length
                          ? result.claim_summary.billing_summary.categories_detected
                              .map((c) => c.replaceAll("_", " "))
                              .join(", ")
                          : "None"
                      }
                    </div>

                    <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 6 }}>
                      Non-Payable Total: ₹{Number(result.claim_summary.billing_summary.non_payable_total || 0).toLocaleString("en-IN")}
                    </div>

                    <div style={{ color: "#e2e8f0", fontWeight: 600, fontSize: 13, marginTop: 10, marginBottom: 6 }}>
                      Deduction Reasons
                    </div>

                    {result.claim_summary.billing_summary.deduction_reason_summary?.length ? (
                      <ul style={{ margin: 0, paddingLeft: 18, color: "#94a3b8", fontSize: 13 }}>
                        {result.claim_summary.billing_summary.deduction_reason_summary.map((reason, idx) => (
                          <li key={idx} style={{ marginBottom: 4 }}>{reason}</li>
                        ))}
                      </ul>
                    ) : (
                      <div style={{ color: "#94a3b8", fontSize: 13 }}>
                        No billing summary available.
                      </div>
                    )}
                  </div>
                )}



                {result.claim_summary.billing_breakdown &&
                Object.keys(result.claim_summary.billing_breakdown).length > 0 ? (
                <div style={{ display: "grid", gap: 12 }}>
                {Object.entries(result.claim_summary.billing_breakdown).map(([category, details]) => (
                <div
                key={category}
                style={{
                padding: 14,
                borderRadius: 10,
                background: "rgba(30,41,59,0.55)",
                border: "1px solid rgba(148,163,184,0.15)"
                }}
                >
                <div style={{ color: "#e2e8f0", fontWeight: 700, marginBottom: 8 }}>
                {category.replaceAll("_", " ").toUpperCase()}
                </div>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>
                Claimed: ₹{Number(details.claimed || 0).toLocaleString("en-IN")}
                </div>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>
                Approved: ₹{Number(details.approved || 0).toLocaleString("en-IN")}
                </div>
                <div style={{
                color:
                details.status === "partial"
                ? "#f59e0b"
                : details.status === "rejected"
                ? "#ef4444"
                : "#10b981",
                fontSize: 13,
                marginTop: 4
                }}>
                Status: {details.status || "unknown"}
                </div>
                {details.reason && (
                <div style={{ color: "#64748b", fontSize: 12, marginTop: 6 }}>
                {details.reason}
                </div>
                )}
                </div>
                ))}
                </div>
                ) : (
                <div style={{ color: "#94a3b8", fontSize: 13 }}>
                No billing breakdown available.
                </div>
                )}
                </div>
                )}


                {/* */}
                {Object.entries(result.claim_summary || {})
                   // added billing relevant emelents to be rendered
                  .filter(([key]) => ![
                          'status',
                          'missing_fields',
                          'action_required',
                          'claim_id',
                          'billing_anomaly_score',
                          'billing_deductions',
                          'billing_breakdown',
                          'billing_summary'
                        ].includes(key))
                  .map(([key, value]) => {
                    // Skip null/undefined values
                    if (value === null || value === undefined) return null;
                    
                    return (
                      <div key={key} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: "1px solid rgba(148,163,184,0.06)" }}>
                        <span style={{ color: "#64748b", fontSize: 13, fontFamily: "mono", textTransform: "uppercase" }}>{key.replace(/_/g, " ")}</span>
                        <span style={{ color: "#e2e8f0", fontSize: 13, fontFamily: "mono", fontWeight: 500 }}>
                          {typeof value === "number" ? (key.includes("amount") ? `₹${value.toLocaleString("en-IN")}` : value.toFixed(2)) : String(value)}
                        </span>
                      </div>
                    );
                  })
                }
                <div style={{ marginTop: 20, padding: 16, background: "rgba(0,0,0,0.3)", borderRadius: 8 }}>
                  <div style={{ color: "#64748b", fontSize: 12, marginBottom: 8 }}>Decision Reasons:</div>
                  {(result.decision_reasons || []).map((r, i) => (
                    <div key={i} style={{ color: "#e2e8f0", fontSize: 13, padding: "3px 0" }}>• {r}</div>
                  ))}
                </div>
              </div>
            )}

            {/* Reasoning Trace Tab */}
            {activeTab === "trace" && (
              <div style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(139,92,246,0.2)", borderRadius: 12, padding: 24, animation: "fadeSlideIn 0.4s ease both" }}>
                <h3 style={{ color: "#8b5cf6", fontSize: 16, fontWeight: 700, marginBottom: 16, fontFamily: "mono" }}>🎯 Full Orchestrator Reasoning Trace</h3>
                <div style={{ fontFamily: "mono", fontSize: 13, lineHeight: 2 }}>
                  {(result.reasoning_trace || []).map((log, i) => (
                    <div key={i} style={{ color: "#a5b4fc", padding: "2px 0" }}>
                      <span style={{ color: "#475569", marginRight: 8 }}>[{String(i).padStart(2, "0")}]</span>{log}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer style={{ borderTop: "1px solid rgba(148,163,184,0.06)", padding: "16px 32px", textAlign: "center", color: "#334155", fontSize: 12, fontFamily: "mono" }}>
        ClaimFlow AI • Multi-Agent Insurance Processing • Backend: localhost:8000 • Frontend: localhost:5173
      </footer>
    </div>
  );
}
