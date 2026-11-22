export interface PdfTextItemLike {
  str: string;
  transform: number[];
  width: number;
  height: number;
}

export interface HighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface MatchedItemRange {
  itemIndex: number;
  startOffset: number;
  endOffset: number;
}

/**
 * Finds which PDF text items (and character offsets within them) match `targetSnippet`.
 * Handles line-break hyphenation (e.g. "co- efficient" vs "coefficient") and whitespace differences.
 */
export function findMatchingTextItems(
  items: PdfTextItemLike[],
  targetSnippet: string
): MatchedItemRange[] {
  if (!targetSnippet.trim() || items.length === 0) return [];

  // Build a stripped alphanumeric stream (lowercase letters & digits only)
  // with a mapping from each stripped character index to (itemIndex, charOffsetInItem).
  const charMap: { itemIndex: number; charIndex: number }[] = [];
  const normChars: string[] = [];

  items.forEach((item, itemIndex) => {
    const str = item.str || '';
    for (let i = 0; i < str.length; i++) {
      const ch = str[i].toLowerCase();
      if (/[a-z0-9]/.test(ch)) {
        normChars.push(ch);
        charMap.push({ itemIndex, charIndex: i });
      }
    }
  });

  const pageNorm = normChars.join('');
  if (!pageNorm) return [];

  const targetNorm = targetSnippet.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (!targetNorm) return [];

  let matchStart = pageNorm.indexOf(targetNorm);
  let matchLen = targetNorm.length;

  // Fallback: if exact full alphanumeric string isn't found (e.g. due to ligatures or footnote numbers),
  // try matching prefix/suffix chunks or longest contiguous alphanumeric window >= 18 chars.
  if (matchStart === -1 && targetNorm.length > 24) {
    const windowSizes = [
      Math.floor(targetNorm.length * 0.75),
      Math.floor(targetNorm.length * 0.5),
      28,
      18,
    ];
    for (const size of windowSizes) {
      if (size >= targetNorm.length || size < 12) continue;
      for (let offset = 0; offset <= targetNorm.length - size; offset += 6) {
        const sub = targetNorm.slice(offset, offset + size);
        const idx = pageNorm.indexOf(sub);
        if (idx !== -1) {
          matchStart = idx;
          matchLen = size;
          break;
        }
      }
      if (matchStart !== -1) break;
    }
  }

  if (matchStart === -1) return [];

  const matchEnd = matchStart + matchLen - 1;
  const rangesByItem = new Map<number, { startOffset: number; endOffset: number }>();

  for (let idx = matchStart; idx <= matchEnd; idx++) {
    const mapped = charMap[idx];
    if (!mapped) continue;
    const existing = rangesByItem.get(mapped.itemIndex);
    if (!existing) {
      rangesByItem.set(mapped.itemIndex, {
        startOffset: mapped.charIndex,
        endOffset: mapped.charIndex + 1,
      });
    } else {
      existing.startOffset = Math.min(existing.startOffset, mapped.charIndex);
      existing.endOffset = Math.max(existing.endOffset, mapped.charIndex + 1);
    }
  }

  const results: MatchedItemRange[] = [];
  rangesByItem.forEach((range, itemIndex) => {
    results.push({
      itemIndex,
      startOffset: range.startOffset,
      endOffset: range.endOffset,
    });
  });

  return results;
}

/**
 * Converts matched PDF text item ranges into viewport pixel rectangles.
 */
export function computeHighlightRects(
  items: PdfTextItemLike[],
  matches: MatchedItemRange[],
  convertToViewportRectangle: (rect: number[]) => number[]
): HighlightRect[] {
  const rawRects: HighlightRect[] = [];

  for (const match of matches) {
    const item = items[match.itemIndex];
    if (!item || !item.str) continue;

    const strLen = Math.max(item.str.length, 1);
    const startRatio = Math.max(0, Math.min(1, match.startOffset / strLen));
    const endRatio = Math.max(startRatio, Math.min(1, match.endOffset / strLen));

    const x = item.transform[4];
    const y = item.transform[5];
    const fontHeight =
      item.height > 0
        ? item.height
        : Math.hypot(item.transform[2], item.transform[3]) || 10;
    const totalWidth = item.width > 0 ? item.width : strLen * (fontHeight * 0.5);

    const xMin = x + totalWidth * startRatio;
    const xMax = x + totalWidth * endRatio;
    const yMin = y - fontHeight * 0.22;
    const yMax = y + fontHeight * 0.92;

    const [vx1, vy1, vx2, vy2] = convertToViewportRectangle([xMin, yMin, xMax, yMax]);
    const left = Math.min(vx1, vx2);
    const top = Math.min(vy1, vy2);
    const width = Math.max(4, Math.abs(vx2 - vx1));
    const height = Math.max(8, Math.abs(vy2 - vy1));

    rawRects.push({ left, top, width, height });
  }

  // Merge adjacent rects on the same horizontal line to prevent visual seams
  const merged: HighlightRect[] = [];
  for (const r of rawRects) {
    const prev = merged[merged.length - 1];
    if (
      prev &&
      Math.abs(prev.top - r.top) < prev.height * 0.45 &&
      r.left <= prev.left + prev.width + 12 &&
      r.left >= prev.left - 4
    ) {
      const newLeft = Math.min(prev.left, r.left);
      const newRight = Math.max(prev.left + prev.width, r.left + r.width);
      prev.left = newLeft;
      prev.width = newRight - newLeft;
      prev.top = Math.min(prev.top, r.top);
      prev.height = Math.max(prev.height, r.height);
    } else {
      merged.push({ ...r });
    }
  }

  return merged;
}
