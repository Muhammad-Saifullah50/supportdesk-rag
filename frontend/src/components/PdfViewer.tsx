import React, { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  FileText,
  Highlighter,
  Info,
  Loader2,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { Citation } from '../types';
import {
  computeHighlightRects,
  findMatchingTextItems,
  HighlightRect,
  PdfTextItemLike,
} from '../utils/pdfHighlight';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/legacy/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

interface PdfViewerProps {
  citation: Citation;
  onClose: () => void;
}

export const PdfViewer: React.FC<PdfViewerProps> = ({ citation, onClose }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const firstHighlightRef = useRef<HTMLDivElement | null>(null);

  const [pdfDoc, setPdfDoc] = useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(citation.page || 1);
  const [numPages, setNumPages] = useState<number>(0);
  const [scale, setScale] = useState<number>(1.2);
  const [loadingDoc, setLoadingDoc] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [highlightRects, setHighlightRects] = useState<HighlightRect[]>([]);
  const [pageDimensions, setPageDimensions] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });
  const [matchedPage, setMatchedPage] = useState<number | null>(null);

  // Load PDF document when pdf_url changes
  useEffect(() => {
    let isCancelled = false;
    setLoadingDoc(true);
    setError(null);
    setHighlightRects([]);
    setMatchedPage(null);

    const loadingTask = pdfjsLib.getDocument(citation.pdf_url);

    loadingTask.promise
      .then(async (doc) => {
        if (isCancelled) return;
        setPdfDoc(doc);
        setNumPages(doc.numPages);

        const targetPage = Math.max(1, Math.min(citation.page || 1, doc.numPages));

        // Verify if the highlight text is on targetPage; if not, search other pages
        let resolvedPage = targetPage;
        if (citation.highlight_text) {
          const pageObj = await doc.getPage(targetPage);
          const tc = await pageObj.getTextContent();
          const items = tc.items.filter((it): it is any => 'str' in it) as PdfTextItemLike[];
          const initialMatches = findMatchingTextItems(items, citation.highlight_text);

          if (initialMatches.length === 0) {
            for (let p = 1; p <= doc.numPages; p++) {
              if (p === targetPage) continue;
              const candidatePage = await doc.getPage(p);
              const candidateTc = await candidatePage.getTextContent();
              const candidateItems = candidateTc.items.filter(
                (it): it is any => 'str' in it
              ) as PdfTextItemLike[];
              if (findMatchingTextItems(candidateItems, citation.highlight_text).length > 0) {
                resolvedPage = p;
                break;
              }
            }
          }
        }

        if (!isCancelled) {
          setMatchedPage(resolvedPage);
          setCurrentPage(resolvedPage);
          setLoadingDoc(false);
        }
      })
      .catch((err) => {
        if (isCancelled) return;
        setError(err?.message || 'Failed to load PDF document.');
        setLoadingDoc(false);
      });

    return () => {
      isCancelled = true;
      loadingTask.destroy();
    };
  }, [citation.pdf_url, citation.page, citation.highlight_text]);

  // Render active page and compute highlight boxes
  useEffect(() => {
    if (!pdfDoc || loadingDoc || !canvasRef.current) return;

    let isCancelled = false;
    let renderTask: pdfjsLib.RenderTask | null = null;

    const renderPage = async () => {
      try {
        const page = await pdfDoc.getPage(currentPage);
        if (isCancelled || !canvasRef.current) return;

        const viewport = page.getViewport({ scale });
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * dpr);
        canvas.height = Math.floor(viewport.height * dpr);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;

        setPageDimensions({
          width: Math.floor(viewport.width),
          height: Math.floor(viewport.height),
        });

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        renderTask = page.render({
          canvasContext: ctx,
          viewport,
        });

        await renderTask.promise;
        if (isCancelled) return;

        // Extract text content for highlighting
        const textContent = await page.getTextContent();
        if (isCancelled) return;

        const items = textContent.items.filter(
          (item): item is any => 'str' in item
        ) as PdfTextItemLike[];

        const matches = findMatchingTextItems(items, citation.highlight_text);
        const rects = computeHighlightRects(items, matches, (rect) =>
          viewport.convertToViewportRectangle(rect)
        );

        setHighlightRects(rects);
      } catch (err: any) {
        if (err?.name !== 'RenderingCancelledException' && !isCancelled) {
          console.error('PDF page render error:', err);
        }
      }
    };

    renderPage();

    return () => {
      isCancelled = true;
      if (renderTask) {
        renderTask.cancel();
      }
    };
  }, [pdfDoc, currentPage, scale, citation.highlight_text, loadingDoc]);

  // Scroll first highlight into view when highlights appear
  useEffect(() => {
    if (highlightRects.length > 0 && firstHighlightRef.current) {
      const timer = setTimeout(() => {
        firstHighlightRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });
      }, 80);
      return () => clearTimeout(timer);
    }
  }, [highlightRects]);

  const scrollToHighlight = () => {
    if (matchedPage && currentPage !== matchedPage) {
      setCurrentPage(matchedPage);
      return;
    }
    firstHighlightRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  };

  return (
    <div className="flex h-full flex-col border-l border-slate-200 bg-white shadow-xl">
      {/* Top Toolbar */}
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-lime-900/50 text-lime-300">
            <FileText className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <h3
              className="truncate text-sm font-semibold text-slate-900"
              title={citation.pdf_filename}
            >
              {citation.pdf_filename}
            </h3>
            <p className="truncate text-xs text-slate-500">
              Section: <span className="font-medium text-slate-700">{citation.section}</span> ·
              Citation [{citation.id}]
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <a
            href={`${citation.pdf_url}#page=${currentPage}`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
            title="Open raw PDF in new tab"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            <span>Open</span>
          </a>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-200 hover:text-slate-800"
            aria-label="Close PDF Viewer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="pdf-demo-disclaimer">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <p>
          <span>Portfolio demo only.</span> This PDF demonstrates source tracing and citation
          highlighting. Never expose real company policies or internal documents in a public portfolio.
        </p>
      </div>

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-4 py-2 text-xs">
        {/* Page Navigation */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage <= 1 || loadingDoc}
            className="rounded border border-slate-200 p-1 text-slate-700 hover:bg-slate-100 disabled:opacity-40"
            title="Previous Page"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="px-2 font-medium text-slate-700">
            Page {currentPage} of {numPages || '…'}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
            disabled={currentPage >= numPages || loadingDoc}
            className="rounded border border-slate-200 p-1 text-slate-700 hover:bg-slate-100 disabled:opacity-40"
            title="Next Page"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        {/* Zoom + Jump to Highlight */}
        <div className="flex items-center gap-2">
          <button
            onClick={scrollToHighlight}
            className="inline-flex items-center gap-1 rounded-md bg-lime-900/50 px-2.5 py-1 font-medium text-lime-200 hover:bg-lime-800/60"
            title="Scroll to highlighted citation text"
          >
            <Highlighter className="h-3.5 w-3.5" />
            <span>
              {highlightRects.length > 0
                ? 'Highlighted'
                : matchedPage
                  ? `Go to p.${matchedPage}`
                  : 'Highlight'}
            </span>
          </button>

          <div className="flex items-center rounded border border-slate-200">
            <button
              onClick={() => setScale((s) => Math.max(0.7, Number((s - 0.15).toFixed(2))))}
              className="p-1 text-slate-700 hover:bg-slate-100"
              title="Zoom Out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span className="border-x border-slate-200 px-2 font-mono text-[11px] text-slate-600">
              {Math.round(scale * 100)}%
            </span>
            <button
              onClick={() => setScale((s) => Math.min(2.2, Number((s + 0.15).toFixed(2))))}
              className="p-1 text-slate-700 hover:bg-slate-100"
              title="Zoom In"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Matched Snippet Banner */}
      <div className="pdf-highlight-banner px-4 py-2.5 text-xs">
        <div className="flex items-start gap-2">
          <Highlighter className="mt-0.5 h-3.5 w-3.5 shrink-0 text-lime-300" />
          <p className="line-clamp-2 leading-relaxed">
            <span className="font-semibold">Highlighted excerpt: </span>"{citation.highlight_text}"
          </p>
        </div>
      </div>

      {/* PDF Canvas & Highlight Overlay Area */}
      <div
        ref={scrollContainerRef}
        className="relative flex-1 overflow-auto bg-slate-100 p-4"
      >
        {loadingDoc && (
          <div className="flex h-64 flex-col items-center justify-center gap-2 text-slate-500">
            <Loader2 className="h-6 w-6 animate-spin text-slate-700" />
            <span className="text-xs">Loading PDF and locating citation text…</span>
          </div>
        )}

        {error && (
          <div className="mx-auto max-w-md rounded-lg border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700">
            {error}
          </div>
        )}

        {!loadingDoc && !error && (
          <div className="mx-auto flex justify-center">
            <div
              className="relative bg-white shadow-md ring-1 ring-slate-300"
              style={{
                width: pageDimensions.width || 'auto',
                height: pageDimensions.height || 'auto',
              }}
            >
              <canvas ref={canvasRef} className="block" />

              {/* Highlight Overlays */}
              <div className="pointer-events-none absolute inset-0">
                {highlightRects.map((rect, idx) => (
                  <div
                    key={idx}
                    ref={idx === 0 ? firstHighlightRef : undefined}
                    data-testid="pdf-highlight-box"
                    className="pdf-citation-highlight absolute rounded-[2px] transition-all duration-300"
                    style={{
                      left: `${rect.left - 2}px`,
                      top: `${rect.top - 1}px`,
                      width: `${rect.width + 4}px`,
                      height: `${rect.height + 2}px`,
                      mixBlendMode: 'multiply',
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
