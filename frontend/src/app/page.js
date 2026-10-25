"use client";

import ReactMarkdown from "react-markdown";

import { useState, useRef, useEffect } from "react";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // keep in sync with app.py
const MAX_QUESTION_CHARS = 1000; // keep in sync with app.py
const MAX_HISTORY_TURNS = 3; // previous Q&A pairs sent for follow-up questions
const MAX_HISTORY_ANSWER_CHARS = 1500;
const WAKE_BANNER_DELAY_MS = 1500; // only show the banner if the server is actually asleep

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const [sessionId, setSessionId] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const [showScrollButton, setShowScrollButton] = useState(false);

  // "checking" -> "waking" (slow cold start) -> "ready", or "down" if unreachable
  const [serverStatus, setServerStatus] = useState("checking");

  const chatEndRef = useRef(null);

  useEffect(() => {
    // Wake the free-tier backend as soon as the page opens, so it's warm by
    // the time the user has typed their first question.
    let cancelled = false;
    const bannerTimer = setTimeout(() => {
      if (!cancelled) setServerStatus((s) => (s === "checking" ? "waking" : s));
    }, WAKE_BANNER_DELAY_MS);

    fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`)
      .then((res) => {
        if (!cancelled) setServerStatus(res.ok ? "ready" : "down");
      })
      .catch(() => {
        if (!cancelled) setServerStatus("down");
      })
      .finally(() => clearTimeout(bannerTimer));

    return () => {
      cancelled = true;
      clearTimeout(bannerTimer);
    };
  }, []);

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

    // Recent completed turns, so follow-ups like "what about options?" make sense
    const history = messages
      .filter((m) => !m.error && !m.streaming && m.answer)
      .slice(-MAX_HISTORY_TURNS)
      .map((m) => ({
        question: m.question,
        answer: m.answer.slice(0, MAX_HISTORY_ANSWER_CHARS),
      }));
  
    setMessages((prev) => [
      ...prev,
      { question: currentQuestion, answer: "", sources: [], streaming: true },
    ]);
  
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, session_id: sessionId, history }),
      });

      if (res.status === 410) {
        // The uploaded document's session expired or the backend restarted
        const data = await res.json().catch(() => ({}));
        clearUploadedDocument();
        setUploadError(data.detail || "Your uploaded document has expired. Please upload it again.");
        throw new Error("Session expired");
      }

      if (res.status === 429) {
        const data = await res.json().catch(() => ({}));
        const err = new Error("Rate limited");
        err.userMessage = data.detail || "You're asking questions too quickly. Please wait a moment and try again.";
        throw err;
      }

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
          answer: err.userMessage || "Something went wrong reaching the assistant. Please try again in a moment.",
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
    e.target.value = ""; // allow re-selecting the same file after an error
    if (!file) return;

    setUploadError(null);

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadError("File is too large. The limit is 10 MB.");
      return;
    }

    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        setUploadError(data.detail || "Upload failed. Please try again.");
        return;
      }

      setSessionId(data.session_id);
      setUploadedFileName(data.filename);
    } catch (err) {
      console.error(err);
      setUploadError("Couldn't reach the server. Please try again in a moment.");
    } finally {
      setUploading(false);
    }
  }
  
  function startNewChat() {
    setMessages([]);
    setQuestion("");
  }

  function clearUploadedDocument() {
    setSessionId(null);
    setUploadedFileName(null);
  }

  return (
    <div className="container">
      <div className="header-row">
        <h1>SEBI Trading Compliance Assistant</h1>
        {messages.length > 0 && (
          <button className="new-chat-btn" onClick={startNewChat} disabled={loading}>
            New chat
          </button>
        )}
      </div>

      {serverStatus === "waking" && (
        <div className="server-banner" role="status">
          Waking up the server. This can take up to a minute on the free tier, and you can type your question meanwhile.
        </div>
      )}
      {serverStatus === "down" && (
        <div className="server-banner server-banner-error" role="alert">
          The server isn&apos;t responding right now. Please try again in a few minutes.
        </div>
      )}

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
              {msg.sources.length > 0 && (
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
              )}
            </div>
          </div>
        ))}

        {loading && <div className="assistant-bubble loading">Thinking...</div>}
        <div ref={chatEndRef} />
      </div>

      <div className="input-area">
        {uploadError && (
          <div className="upload-error" role="alert">
            <span>{uploadError}</span>
            <button onClick={() => setUploadError(null)} className="clear-doc-btn" aria-label="Dismiss error">
              ✕
            </button>
          </div>
        )}

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
              maxLength={MAX_QUESTION_CHARS}
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