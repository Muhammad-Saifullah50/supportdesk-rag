"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, ArrowUp, BookOpen, FileText, Loader2, Sparkles } from 'lucide-react';
import { AskResponse, ChatMessage, Citation } from './types';
import { parseAnswerSegments } from './utils/citationParser';
import { PdfViewer } from './components/PdfViewer';

const SAMPLE_PROMPTS = [
  'What are the current plan prices, monthly request quotas, and per-project rate limits?',
  'How did Starter pricing and included requests change on 2026-01-01 compared with the archived 2025 plan?',
  'What is the difference between rate_limited and quota_exceeded, and what should I do for each?',
  'Which project regions are available, and can the region be changed later?',
  'What is the refund window, and how are annual plan cancellations prorated?',
  'How do I verify a webhook signature, and when should I reject its timestamp?',
  'What happens to my account and data if a payment remains unpaid?',
  'How do I export my data, and how long is the download link valid?',
  'What are the Enterprise Sev1 response targets with and without Premier Support?',
  'How long are audit logs retained on the Growth and Enterprise plans?',
];

interface ChatTurn {
  question: string;
  response: AskResponse | null;
  error: string | null;
}

export const AssistantPage: React.FC = () => {
  const [question, setQuestion] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  const submitQuestion = async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed || loading) return;

    setQuestion('');
    setLoading(true);
    setTurns((currentTurns) => [...currentTurns, { question: trimmed, response: null, error: null }]);
    setActiveCitation(null);
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, history: history.slice(-20) }),
      });

      if (res.status === 429) {
        setTurns((currentTurns) => currentTurns.map((turn, index) => (
          index === currentTurns.length - 1
            ? { ...turn, error: "You've reached today's demo limit of 10 questions per IP. Try again after midnight UTC." }
            : turn
        )));
        return;
      }

      if (!res.ok) {
        const requestId = res.headers.get('x-request-id');
        const body = await res.json().catch(() => null) as { detail?: string } | null;
        const detail = body?.detail || `HTTP ${res.status}`;
        throw new Error(requestId ? `${detail} (request ${requestId})` : detail);
      }
      const data: AskResponse = await res.json();
      setTurns((currentTurns) => currentTurns.map((turn, index) => (
        index === currentTurns.length - 1 ? { ...turn, response: data } : turn
      )));
      setHistory((messages) => [
        ...messages,
        { role: 'user' as const, content: trimmed },
        { role: 'assistant' as const, content: data.answer },
      ].slice(-20));
    } catch (error) {
      setTurns((currentTurns) => currentTurns.map((turn, index) => (
        index === currentTurns.length - 1
          ? {
              ...turn,
              error: error instanceof Error
                ? error.message
                : 'The grounded retrieval service is unavailable. Try again later.',
            }
          : turn
      )));
    } finally {
      setLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitQuestion(question);
  };

  return (
    <div className="assistant-workspace">
      <section className={`assistant-conversation ${activeCitation ? 'is-split' : ''}`}>
        <header className="assistant-header">
          <div className="assistant-title-lockup">
            <div className="assistant-brand-icon"><BookOpen size={17} /></div>
            <div className="assistant-title-copy">
              <h1>SupportDesk-RAG</h1>
              <p>Grounded retrieval</p>
            </div>
          </div>
          <Link href="/" className="assistant-back-link"><ArrowLeft size={15} /> Overview</Link>
        </header>

        <main className="assistant-content">
          <div className="assistant-thread">
            {turns.length === 0 && (
              <section className="assistant-welcome">
                <div className="welcome-mark"><Sparkles size={23} /></div>
                <h2>Ask the corpus.</h2>
                <p>Answers are grounded in the documents, with sources you can inspect.</p>
                <div className="sample-grid" aria-label="Sample questions">
                  {SAMPLE_PROMPTS.map((sample) => (
                    <button
                      key={sample}
                      type="button"
                      onClick={() => {
                        setQuestion(sample);
                        submitQuestion(sample);
                      }}
                      className="sample-prompt"
                    >
                      <span>{sample}</span>
                      <ArrowUp size={15} />
                    </button>
                  ))}
                </div>
              </section>
            )}

            {turns.map((turn, turnIndex) => {
              const segments = turn.response
                ? parseAnswerSegments(turn.response.answer, turn.response.citations)
                : [];

              return (
                <div className="conversation-turn" key={`${turnIndex}-${turn.question}`}>
                  <div className="user-message">
                    <p>{turn.response?.question ?? turn.question}</p>
                  </div>

                  {turn.response && (
                    <article className="assistant-message answer-message">
                      <div className="assistant-message-mark"><Sparkles size={17} /></div>
                      <div className="answer-body">
                        <p className="answer-copy">
                          {segments.map((seg, idx) => {
                            if (seg.type === 'text') {
                              return <React.Fragment key={idx}>{seg.content}</React.Fragment>;
                            }

                            const isSelected = activeCitation === seg.citation;

                            return (
                              <button
                                key={idx}
                                type="button"
                                onClick={() => seg.citation && setActiveCitation(seg.citation)}
                                aria-label={seg.citation
                                  ? `Open citation ${seg.displayId}: ${seg.citation.pdf_filename}, page ${seg.citation.page}`
                                  : `Open citation ${seg.displayId}`}
                                title={seg.citation
                                  ? `Citation ${seg.displayId}: ${seg.citation.pdf_filename} (Page ${seg.citation.page})`
                                  : seg.raw}
                                className={`citation-badge ${isSelected ? 'is-selected' : ''}`}
                              >
                                <FileText size={12} />
                              </button>
                            );
                          })}
                        </p>

                        {turn.response.citations.length > 0 && (
                          <div className="citation-list">
                            <p className="citation-list-heading">Sources</p>
                            {turn.response.citations.map((cit, citationIndex) => (
                              <button
                                key={`${cit.id}-${citationIndex}`}
                                type="button"
                                onClick={() => setActiveCitation(cit)}
                                className={`citation-source ${activeCitation === cit ? 'is-selected' : ''}`}
                              >
                                <span className="citation-source-id">{cit.id}</span>
                                <span className="citation-source-copy">
                                  <strong>{cit.pdf_filename}</strong>
                                  <span>Page {cit.page} · {cit.section}</span>
                                  <span className="citation-excerpt">“{cit.highlight_text}”</span>
                                </span>
                                <ArrowUp size={13} className="citation-source-arrow" />
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </article>
                  )}

                  {turnIndex === turns.length - 1 && loading && (
                    <div className="assistant-message assistant-loading" role="status">
                      <div className="assistant-message-mark"><Sparkles size={17} /></div>
                      <Loader2 size={16} className="loading-spinner" />
                      <span>Checking the source documents…</span>
                    </div>
                  )}

                  {turn.error && <p role="alert" className="assistant-error">{turn.error}</p>}
                </div>
              );
            })}
          </div>
        </main>

        <div className="assistant-composer-wrap">
          <form onSubmit={handleFormSubmit} className="assistant-composer">
            <label htmlFor="question-input" className="sr-only">Ask a question</label>
            <textarea
              id="question-input"
              rows={1}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  submitQuestion(question);
                }
              }}
              placeholder="Ask a question about the sample documents"
              className="question-input ring-0"
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="composer-submit"
              aria-label="Send question"
            >
              {loading ? <Loader2 size={17} className="loading-spinner" /> : <ArrowUp size={18} />}
            </button>
          </form>
          <p className="composer-note">Grounded answers only · Press Enter to ask, Shift + Enter for a new line</p>
        </div>
      </section>

      {activeCitation && (
        <div className="assistant-pdf-pane">
          <PdfViewer
            citation={activeCitation}
            onClose={() => setActiveCitation(null)}
          />
        </div>
      )}
    </div>
  );
};

export default AssistantPage;
