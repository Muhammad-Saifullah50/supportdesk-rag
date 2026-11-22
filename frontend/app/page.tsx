import Link from 'next/link';
import {
  ArrowDownRight,
  ArrowRight,
  BookOpenCheck,
  FileSearch,
  Layers3,
  ListFilter,
  Search,
  ShieldCheck,
} from 'lucide-react';

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <BookOpenCheck size={19} strokeWidth={2.2} />
    </span>
  );
}

export default function HomePage() {
  return (
    <main className="landing-page">
      <header className="site-header">
        <Link className="brand-lockup" href="/" aria-label="SupportDesk-RAG project home">
          <BrandMark />
          <span>SupportDesk-RAG</span>
          <span className="project-tag">PROJECT</span>
        </Link>
        <nav className="site-nav" aria-label="Project navigation">
          <a href="#approach">Approach</a>
          <a href="#pipeline">Pipeline</a>
        </nav>
        <Link className="header-cta" href="/assistant">
          Open assistant <ArrowRight size={15} />
        </Link>
      </header>

      <section className="hero-section">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-dot" /> DOCUMENT Q&A · GROUNDED RETRIEVAL</div>
          <h1>Answers from<br /><em>the evidence.</em></h1>
          <p className="hero-description">
            Ask questions about the indexed sample corpus, review inline citations, and open the
            source PDFs to inspect the highlighted passages behind an answer.
          </p>
          <div className="hero-actions">
            <Link className="button-primary" href="/assistant">
              Open the assistant <ArrowRight size={17} />
            </Link>
            <a className="text-link" href="#pipeline">
              View retrieval flow <ArrowDownRight size={15} />
            </a>
          </div>
          <div className="hero-note"><span className="eyebrow-dot" /> Indexed sample corpus · PDF citations</div>
        </div>

        <div className="hero-visual" aria-label="Overview of the implemented retrieval flow">
          <div className="visual-orbit orbit-one" />
          <div className="visual-orbit orbit-two" />
          <div className="answer-window pipeline-preview">
            <div className="window-topbar">
              <div className="window-brand"><BrandMark /><span>QUESTION TO ANSWER</span></div>
              <span className="window-status"><i /> GROUNDED FLOW</span>
            </div>
            <div className="trace-row">
              <span className="trace-step"><Search size={14} /></span>
              <span className="trace-copy"><strong>Retrieve</strong><small>Dense search + BM25</small></span>
              <span className="trace-step-name">01</span>
            </div>
            <div className="trace-row">
              <span className="trace-step"><ListFilter size={14} /></span>
              <span className="trace-copy"><strong>Rank evidence</strong><small>Fuse, then rerank top 15</small></span>
              <span className="trace-step-name">02</span>
            </div>
            <div className="trace-row">
              <span className="trace-step"><FileSearch size={14} /></span>
              <span className="trace-copy"><strong>Answer with sources</strong><small>Open citations in the PDF viewer</small></span>
              <span className="trace-step-name">03</span>
            </div>
            <div className="window-footer"><span className="footer-pulse" /> Only retrieved evidence is sent to answer generation</div>
          </div>
          <div className="floating-tag tag-top"><span>01</span> Inline citations</div>
          <div className="floating-tag tag-bottom"><span>02</span> PDF passages highlighted</div>
          <div className="visual-caption">Implemented retrieval and citation flow</div>
        </div>
        <div className="hero-index">01 — PROJECT OVERVIEW</div>
      </section>

      <section className="principles-section" id="approach">
        <div className="section-intro">
          <p className="eyebrow">ANSWER BEHAVIOR</p>
          <h2>Evidence first.<br /><em>Sources in view.</em></h2>
          <p>
            Answers are generated from retrieved document chunks. Each citation links to its source
            PDF and the matching passage can be inspected in context.
          </p>
        </div>
        <div className="principle-list">
          <article className="principle-card">
            <span className="principle-number">01</span>
            <div className="principle-icon"><FileSearch size={20} /></div>
              <div><h3>Open and inspect citations</h3><p>Each citation opens the source PDF at the cited page and highlights the matching text.</p></div>
            <ArrowRight className="principle-arrow" size={17} />
          </article>
          <article className="principle-card">
            <span className="principle-number">02</span>
              <div className="principle-icon"><Layers3 size={20} /></div>
              <div><h3>Conflicts called out</h3><p>The generation prompt instructs the model to identify conflicting retrieved sources and cite each one.</p></div>
            <ArrowRight className="principle-arrow" size={17} />
          </article>
          <article className="principle-card">
            <span className="principle-number">03</span>
            <div className="principle-icon"><ShieldCheck size={20} /></div>
              <div><h3>Refuse without evidence</h3><p>If no retrieved chunks pass the reranking threshold, the API returns the refusal without invoking answer generation.</p></div>
            <ArrowRight className="principle-arrow" size={17} />
          </article>
        </div>
      </section>

      <section className="workflow-section" id="pipeline">
        <div className="workflow-heading">
          <p className="eyebrow">IMPLEMENTED RETRIEVAL FLOW</p>
          <h2>Retrieve, rerank,<br /><em>then answer.</em></h2>
          <p className="workflow-description">
            Answer generation receives only the evidence that survives retrieval and reranking.
          </p>
        </div>
        <div className="workflow-steps">
          <article><span>01</span><h3>Search the corpus</h3><p>Run dense vector search and BM25 over the indexed document chunks.</p></article>
          <div className="step-connector" />
          <article><span>02</span><h3>Fuse candidates</h3><p>Combine the ranked result lists with Reciprocal Rank Fusion.</p></article>
          <div className="step-connector" />
          <article><span>03</span><h3>Rerank evidence</h3><p>Cross-encode the top 15 chunks and discard scores below 0.35.</p></article>
          <div className="step-connector" />
          <article><span>04</span><h3>Answer or refuse</h3><p>Generate from surviving chunks, or return the exact refusal when none remain.</p></article>
        </div>
        <div className="project-stack">
          <span><Layers3 size={14} /> CURRENT STACK</span>
          <p>Next.js · FastAPI · Qdrant · BM25 · Cross-encoder · OpenRouter embeddings</p>
        </div>
        <Link className="workflow-cta" href="/assistant">Ask a question <ArrowRight size={16} /></Link>
      </section>

      <footer className="site-footer">
        <Link className="brand-lockup" href="/">
          <BrandMark /><span>SupportDesk-RAG</span>
        </Link>
        <span>Grounded document retrieval</span>
        <Link href="/assistant">Open assistant <ArrowRight size={14} /></Link>
      </footer>
    </main>
  );
}
