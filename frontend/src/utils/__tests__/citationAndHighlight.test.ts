import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import { parseAnswerSegments } from '../citationParser';
import {
  computeHighlightRects,
  findMatchingTextItems,
  PdfTextItemLike,
} from '../pdfHighlight';
import { Citation } from '../../types';

const sampleCitations: Citation[] = [
  {
    id: '1',
    document_id: 'LungPaper',
    label: 'LungPaper.pdf · p.2',
    pdf_filename: 'LungPaper.pdf',
    pdf_url: '/docs/LungPaper.pdf',
    page: 2,
    section: 'Abstract',
    highlight_text:
      'results in an improved dice coefficient of 0.8472.',
  },
];

describe('parseAnswerSegments', () => {
  it('parses numeric inline citation markers like [1]', () => {
    const segments = parseAnswerSegments(
      'The model achieved 0.8472 [1] on the test set.',
      sampleCitations
    );
    expect(segments).toHaveLength(3);
    expect(segments[0]).toEqual({ type: 'text', content: 'The model achieved 0.8472 ' });
    expect(segments[1]).toMatchObject({
      type: 'citation',
      displayId: '1',
      citation: sampleCitations[0],
    });
    expect(segments[2]).toEqual({ type: 'text', content: ' on the test set.' });
  });

  it('parses spec-format inline citations [Doc: ..., Section: ...]', () => {
    const segments = parseAnswerSegments(
      'Tumor segmentation improved [Doc: LungPaper, Section: Abstract].',
      sampleCitations
    );
    expect(segments).toHaveLength(3);
    expect(segments[1]).toMatchObject({
      type: 'citation',
      citation: sampleCitations[0],
    });
  });

  it('resolves document and section together instead of a generic section match', () => {
    const citations = [
      ...sampleCitations,
      {
        ...sampleCitations[0],
        id: '2',
        document_id: 'OtherPaper',
        pdf_filename: 'OtherPaper.pdf',
      },
    ];
    const segments = parseAnswerSegments(
      'The finding is reported [Doc: OtherPaper, Section: Abstract].',
      citations
    );

    expect(segments[1]).toMatchObject({
      type: 'citation',
      citation: citations[1],
    });
  });
});

describe('findMatchingTextItems & computeHighlightRects', () => {
  it('matches text across line-wrapped hyphenated PDF text items', () => {
    const items: PdfTextItemLike[] = [
      {
        str: 'the model architecture results in an improved dice co-',
        transform: [10, 0, 0, 10, 50, 700],
        width: 240,
        height: 10,
      },
      {
        str: 'efficient of 0.8472. A key characteristic',
        transform: [10, 0, 0, 10, 50, 685],
        width: 200,
        height: 10,
      },
    ];

    const matches = findMatchingTextItems(
      items,
      'results in an improved dice coefficient of 0.8472.'
    );
    expect(matches).toHaveLength(2);
    expect(matches[0].itemIndex).toBe(0);
    expect(matches[1].itemIndex).toBe(1);

    const rects = computeHighlightRects(items, matches, (r) => r);
    expect(rects.length).toBeGreaterThanOrEqual(2);
    expect(rects[0].width).toBeGreaterThan(0);
    expect(rects[1].width).toBeGreaterThan(0);
  });

  it('locates cited text in the current policy PDFs in docs/', async () => {
    const checks = [
      {
        file: path.resolve(
          __dirname,
          '../../../../docs/01_pricing_and_plans (2).pdf'
        ),
        page: 1,
        snippet:
          'Northwind Cloud is a developer platform that provides hosted APIs for document processing, object storage and message queues.',
      },
      {
        file: path.resolve(
          __dirname,
          '../../../../docs/07_changelog (2).pdf'
        ),
        page: 1,
        snippet:
          'Growth plan rate limit raised from 2,400 to 3,000 requests per minute.',
      },
      {
        file: path.resolve(
          __dirname,
          '../../../../docs/05_onboarding_guide (2).pdf'
        ),
        page: 1,
        snippet: 'The region cannot be changed later.',
      },
    ];

    for (const check of checks) {
      const data = new Uint8Array(fs.readFileSync(check.file));
      const doc = await pdfjsLib.getDocument({ data }).promise;
      const page = await doc.getPage(check.page);
      const tc = await page.getTextContent();
      const items = tc.items.filter((it: any) => 'str' in it) as PdfTextItemLike[];
      const matches = findMatchingTextItems(items, check.snippet);
      expect(matches.length).toBeGreaterThan(0);
    }
  });
});
