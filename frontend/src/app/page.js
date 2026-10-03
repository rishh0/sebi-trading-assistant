"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const [sessionId, setSessionId] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [uploading, setUploading] = useState(false);

  async function handleAsk() {
    if (!question.trim()) return;
  
    setLoading(true);
    const currentQuestion = question;
    setQuestion("");
  
    setMessages((prev) => [
      ...prev,
      { question: currentQuestion, answer: "", sources: [], streaming: true },
    ]);
  
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, session_id: sessionId }),
      });
  
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
  
      let fullText = "";       // everything received so far
      let revealedLength = 0;  // how much we've actually shown on screen
      let streamDone = false;
  
      // Reveal a few characters at a steady pace, independent of network speed
      const revealInterval = setInterval(() => {
        if (revealedLength < fullText.length) {
          revealedLength = Math.min(revealedLength + 3, fullText.length);
  
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              answer: fullText.slice(0, revealedLength),
            };
            return updated;
          });
        } else if (streamDone) {
          clearInterval(revealInterval);
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              streaming: false,
            };
            return updated;
          });
          setLoading(false);
        }
      }, 15);
  
      let sourcesData = [];

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  fullText += decoder.decode(value, { stream: true });

  const sourceMatch = fullText.match(/::SOURCES::(.*?)::END_SOURCES::/s);
  if (sourceMatch) {
    sourcesData = JSON.parse(sourceMatch[1]);
    fullText = fullText.replace(/::SOURCES::.*?::END_SOURCES::/s, "");
  }
}

setMessages((prev) => {
  const updated = [...prev];
  updated[updated.length - 1] = {
    ...updated[updated.length - 1],
    sources: sourcesData,
  };
  return updated;
});
  
      streamDone = true;
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
  
    setUploading(true);
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
  
      setSessionId(data.session_id);
      setUploadedFileName(data.filename);
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
    }
  }
  
  function clearUploadedDocument() {
    setSessionId(null);
    setUploadedFileName(null);
  }

  return (
    <div className="container">
      <h1>SEBI Trading Compliance Assistant</h1>

      <div className="upload-section">
        {uploadedFileName ? (
          <div className="active-document">
            <span>📄 Custom document active: <strong>{uploadedFileName}</strong></span>
            <button onClick={clearUploadedDocument} className="clear-doc-btn">
              Clear
            </button>
          </div>
        ) : (
          <label className="upload-btn">
            {uploading ? "Uploading..." : "+ Upload a PDF to ask about"}
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              disabled={uploading}
              style={{ display: "none" }}
            />
          </label>
        )}
      </div>

      <div className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Try asking:</p>
            <button onClick={() => setQuestion("What is the peak margin rule?")}>
              What is the peak margin rule?
            </button>
            <button onClick={() => setQuestion("How is F&O turnover calculated for tax audit?")}>
              How is F&O turnover calculated for tax audit?
            </button>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className="message-pair">
            <div className="user-bubble">{msg.question}</div>
            <div className="assistant-bubble">
              <div className="bubble-label">Assistant</div>
              <div className="markdown-body">
                <ReactMarkdown>{msg.answer}</ReactMarkdown>
              </div>
              <div className="sources">
                <strong>Sources:</strong>
                <div className="sources-list">
                  {msg.sources.map((src, i) => (
                    <details key={i} className="source-item">
                      <summary>{src.source}</summary>
                      <p className="source-excerpt">{src.excerpt}...</p>
                    </details>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}

        {loading && <div className="assistant-bubble loading">Thinking...</div>}
      </div>

      <div className="input-row">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Ask a question about SEBI rules or F&O taxation..."
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "..." : "Ask"}
        </button>
      </div>
    </div>
  );
}