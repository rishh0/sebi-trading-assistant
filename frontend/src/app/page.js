"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function handleAsk() {
    if (!question.trim()) return;

    setLoading(true);
    const currentQuestion = question;
    setQuestion("");

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion }),
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { question: currentQuestion, answer: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
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
            <div className="assistant-bubble">
              <div className="bubble-label">Assistant</div>
              <p>{msg.answer}</p>
              <div className="sources">
                <strong>Sources:</strong>
                <ul>
                  {msg.sources.map((src, i) => (
                    <li key={i}>{src}</li>
                  ))}
                </ul>
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