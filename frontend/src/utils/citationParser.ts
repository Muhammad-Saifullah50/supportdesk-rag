import { Citation } from '../types';

export type AnswerSegment =
  | { type: 'text'; content: string }
  | { type: 'citation'; raw: string; citation?: Citation; displayId: string };

/**
 * Splits an answer string containing inline citations like `[1]` or
 * `[Doc: <doc_id>, Section: <section>]` into renderable segments.
 */
export function parseAnswerSegments(
  answer: string,
  citations: Citation[]
): AnswerSegment[] {
  if (!answer) return [];

  const citationById = new Map<string, Citation>();
  citations.forEach((c) => {
    citationById.set(c.id.toLowerCase(), c);
  });

  // Match either [1] / [2] or [Doc: ..., Section: ...]
  const pattern = /\[(\d+)\]|\[Doc:\s*([^,\]]+),\s*Section:\s*([^\]]+)\]/g;
  const segments: AnswerSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(answer)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: 'text',
        content: answer.slice(lastIndex, match.index),
      });
    }

    if (match[1]) {
      const id = match[1];
      const found = citationById.get(id.toLowerCase());
      segments.push({
        type: 'citation',
        raw: match[0],
        citation: found,
        displayId: id,
      });
    } else {
      const docId = match[2].trim();
      const section = match[3].trim();
      const found =
        citations.find(
          (c) =>
            c.document_id.toLowerCase() === docId.toLowerCase() &&
            c.section.toLowerCase() === section.toLowerCase()
        ) ||
        citations.find(
          (c) =>
            c.pdf_filename.toLowerCase().includes(docId.toLowerCase()) &&
            c.section.toLowerCase() === section.toLowerCase()
        );
      segments.push({
        type: 'citation',
        raw: match[0],
        citation: found,
        displayId: `${docId} · ${section}`,
      });
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < answer.length) {
    segments.push({
      type: 'text',
      content: answer.slice(lastIndex),
    });
  }

  return segments;
}
