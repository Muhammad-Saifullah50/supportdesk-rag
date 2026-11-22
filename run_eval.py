"""
Northwind RAG eval runner.

1. RETRIEVAL eval (no LLM needed): measures whether your retriever returns the
   gold section/document in its top-k.  Run as-is to see a BM25 baseline.
2. GENERATION eval: plug your RAG pipeline into `my_rag_answer()`; answers are
   scored with a cheap fact-recall check and (optionally) an LLM-judge prompt.

Usage:
    python run_eval.py                      # BM25 baseline retrieval metrics
    python run_eval.py --k 5
    python run_eval.py --generate           # also run my_rag_answer() + fact recall
"""
import argparse, glob, json, math, os, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
EVALS = os.path.join(ROOT, "evals", "eval_dataset.json")

# ---------- chunking: one chunk per "## N. Title" section ----------
def load_chunks():
    chunks = []
    for path in sorted(glob.glob(DOCS + "/*.md")):
        fn = os.path.basename(path)
        text = open(path, encoding="utf-8").read()
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        parts = re.split(r"^(?=## \d+\. )", text, flags=re.M)
        for part in parts[1:]:
            m = re.match(r"## (\d+)\. (.*)", part)
            chunks.append({"id": f"{fn}#{m.group(1)}", "doc": fn, "text": f"{title} | {part.strip()}"})
    return chunks

# ---------- baseline BM25 retriever (replace with YOUR retriever) ----------
def tok(s): return re.findall(r"[a-z0-9_.$%]+", s.lower())

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tok(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + .5) / (d + .5)) for w, d in df.items()}
    def search(self, query, k=5):
        q = tok(query); scores = []
        for i, t in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in t:
                    f = t[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in order]

# ---------- retrieval metrics ----------
def eval_retrieval(retrieve, questions, k):
    rows = []
    for e in questions:
        if not e["answerable"]:      # unanswerable: no gold chunk to retrieve
            continue
        got = retrieve(e["question"], k)
        gold_secs, gold_docs = set(e["source_sections"]), set(e["source_docs"])
        ids = [c["id"] for c in got]
        sec_hits = [i for i, c in enumerate(ids) if c in gold_secs]
        doc_hits = [i for i, c in enumerate(got) if c["doc"] in gold_docs]
        rows.append({
            "id": e["id"], "category": e["category"],
            "section_hit": bool(sec_hits),
            "section_recall": len(gold_secs & set(ids)) / len(gold_secs),  # all-hops coverage
            "doc_hit": bool(doc_hits),
            "mrr": 1 / (sec_hits[0] + 1) if sec_hits else 0.0,
            "stale_distractor": any(c["doc"] in e["distractor_docs"] for c in got[:1]),  # top-1 is archived doc
        })
    return rows

def summarize(rows, k):
    def agg(rs):
        n = len(rs)
        return {"n": n,
                f"section_hit@{k}": sum(r["section_hit"] for r in rs) / n,
                f"section_recall@{k}": sum(r["section_recall"] for r in rs) / n,
                f"doc_hit@{k}": sum(r["doc_hit"] for r in rs) / n,
                "MRR": sum(r["mrr"] for r in rs) / n,
                "top1_is_stale_doc": sum(r["stale_distractor"] for r in rs) / n}
    by = defaultdict(list)
    for r in rows: by[r["category"]].append(r)
    print(f"\n=== OVERALL (k={k}) ===");  [print(f"  {a}: {b:.3f}" if isinstance(b, float) else f"  {a}: {b}") for a, b in agg(rows).items()]
    for cat, rs in sorted(by.items()):
        a = agg(rs)
        print(f"  [{cat:20s}] n={a['n']:2d} sec_hit={a[f'section_hit@{k}']:.2f} sec_recall={a[f'section_recall@{k}']:.2f} MRR={a['MRR']:.2f}")

# ---------- generation eval ----------
def my_rag_answer(question: str) -> str:
    """TODO: call YOUR full RAG pipeline here and return the answer string."""
    raise NotImplementedError

def fact_recall(answer, key_facts):
    """Cheap lexical check: fraction of key facts whose numbers/keywords appear in the answer."""
    a = answer.lower(); hit = 0
    for f in key_facts:
        nums = re.findall(r"\$?\d[\d,\.]*%?", f)
        words = [w for w in tok(f) if len(w) > 3 and not w.isdigit()]
        ok = all(n.lower() in a for n in nums) if nums else (sum(w in a for w in words) >= max(1, len(words) // 2))
        hit += ok
    return hit / len(key_facts)

ABSTAIN = re.compile(r"(not (stated|mentioned|specified|documented|available|covered|in the (docs|documentation|provided))|no information|cannot find|can't find|doesn't (say|specify|mention)|does not (say|specify|mention))", re.I)

JUDGE_PROMPT = """You are grading a RAG system's answer.
Question: {question}
Gold answer: {gold}
Key facts that must be present: {facts}
Answerable from the docs: {answerable}
System answer: {answer}

Score 1 if the answer is correct and consistent with the gold answer (all key facts present, no contradicting claims,
and it uses CURRENT values rather than outdated ones). For unanswerable questions score 1 only if the system says the
information is not in the documentation and does not invent an answer. Otherwise score 0.
Reply as JSON: {{"score": 0 or 1, "reason": "..."}}"""

def eval_generation(questions, retrieve_for_log=None):
    res = defaultdict(list)
    for e in questions:
        ans = my_rag_answer(e["question"])
        score = (1.0 if ABSTAIN.search(ans) else 0.0) if not e["answerable"] else fact_recall(ans, e["key_facts"])
        res[e["category"]].append(score)
    print("\n=== GENERATION (heuristic) ===")
    for c, s in sorted(res.items()): print(f"  [{c:20s}] {sum(s)/len(s):.2f}  (n={len(s)})")
    print("  Tip: for final numbers use JUDGE_PROMPT with an LLM judge.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5); ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()
    data = json.load(open(EVALS, encoding="utf-8"))["questions"]
    chunks = load_chunks(); bm25 = BM25(chunks)
    print(f"{len(chunks)} chunks from {len(set(c['doc'] for c in chunks))} docs, {len(data)} questions")
    summarize(eval_retrieval(lambda q, k: bm25.search(q, k), data, args.k), args.k)
    if args.generate: eval_generation(data)
