"use client";

import ReactMarkdown from "react-markdown";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const [sessionId, setSessionId] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [showScrollButton, setShowScrollButton] = useState(false);
  const [atTop, setAtTop] = useState(true);

  const chatEndRef = useRef(null);

  useEffect(() => {
    const distanceFromBottom =
      document.documentElement.scrollHeight -
      window.scrollY -
      window.innerHeight;
  
    if (distanceFromBottom < 150) {
      chatEndRef.current?.scrollIntoView({ behavior: "instant" });
    }
  }, [messages]);

  useEffect(() => {
    function handleScroll() {
      const scrollTop = window.scrollY;
      const scrollHeight = document.documentElement.scrollHeight;
      const clientHeight = window.innerHeight;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
  
      setShowScrollButton(distanceFromBottom > 150);
    }
  
    window.addEventListener("scroll", handleScroll);
    handleScroll();
  
    return () => window.removeEventListener("scroll", handleScroll);
  }, [messages]);

  function handleScrollButtonClick() {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }

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

      if (!res.ok) {
        throw new Error(`Server responded with status ${res.status}`);
      }
  
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
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          answer: "Something went wrong reaching the assistant. Please try again in a moment.",
          streaming: false,
          error: true,
        };
        return updated;
      });
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
            <div className={`assistant-bubble ${msg.error ? "error-bubble" : ""}`}>
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
        <div ref={chatEndRef} />
      </div>

      <div className="input-area">
        {uploadedFileName && (
          <div className="active-document-chip">
            <span className="chip-text">📄 {uploadedFileName}</span>
            <button onClick={clearUploadedDocument} className="clear-doc-btn" aria-label="Remove document">
              ✕
            </button>
          </div>
        )}

        <div className="input-row">
          <div className="input-wrapper">
            <label className="attach-icon-btn" title="Upload a PDF">
              {uploading ? "…" : "+"}
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={uploading}
                style={{ display: "none" }}
              />
            </label>

            <input
              className="main-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              placeholder="Ask a question about SEBI rules or F&O taxation..."
            />
          </div>
          <button onClick={handleAsk} disabled={loading}>
            {loading ? "..." : "Ask"}
          </button>
        </div>
      </div>

      {showScrollButton && (
        <button className="scroll-fab" onClick={handleScrollButtonClick}>
          ↓
        </button>
      )}

    </div>
  );
}