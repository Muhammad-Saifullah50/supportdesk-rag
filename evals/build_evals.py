import json, glob, os, re

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
DOCMAP = {os.path.basename(p)[:2]: os.path.basename(p) for p in glob.glob(DOCS_DIR + "/*.md")}

def S(*refs):
    """S('01:4','03:3') -> ['01_pricing_and_plans.md#4', ...]"""
    out = []
    for r in refs:
        d, s = r.split(":")
        out.append(f"{DOCMAP[d]}#{s}")
    return out

EVALS = []
def q(category, difficulty, question, gold, facts, src, distractors=None, answerable=True, notes=None):
    EVALS.append({
        "id": f"q{len(EVALS)+1:03d}",
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": gold,
        "key_facts": facts,
        "answerable": answerable,
        "source_sections": S(*src),
        "source_docs": sorted({s.split('#')[0] for s in S(*src)}),
        "distractor_docs": [DOCMAP[d] for d in (distractors or [])],
        "notes": notes or "",
    })

# ---------------- SINGLE FACT ----------------
SF = "single_fact"
q(SF,"easy","How much does the Starter plan cost per month?","$29 per month.",["$29/month"],["01:2"],["09"])
q(SF,"easy","How many API requests per month are included in the Growth plan?","5,000,000 requests per month.",["5,000,000"],["01:2"])
q(SF,"easy","What is the API rate limit for the Free plan?","60 requests per minute (per project).",["60 requests/minute"],["03:3"])
q(SF,"easy","What is the base URL of the current API?","https://api.northwindcloud.example/v3",["https://api.northwindcloud.example/v3"],["03:1"])
q(SF,"easy","Which encryption standard does Northwind use for data at rest?","AES-256.",["AES-256"],["04:2"])
q(SF,"easy","Which data residency regions can I choose from?","us-east-1 (Virginia, USA), eu-central-1 (Frankfurt, Germany) and ap-southeast-1 (Singapore).",["us-east-1","eu-central-1","ap-southeast-1"],["04:3"])
q(SF,"easy","How long is the money-back guarantee window for a first paid subscription?","14 days from the first paid charge.",["14 days","first paid charge"],["02:2"],["09"])
q(SF,"easy","How long is the free Growth trial and does it need a credit card?","14 days, and no credit card is required.",["14 days","no credit card"],["01:5"])
q(SF,"easy","Which HTTP header carries the webhook signature?","X-Northwind-Signature.",["X-Northwind-Signature"],["03:9"])
q(SF,"easy","What is the maximum payload size on the Growth plan?","50 MB.",["50 MB"],["03:5"])
q(SF,"easy","What does an HTTP 402 payment_required response mean?","The account is suspended because an invoice is unpaid (after 7 days).",["account suspended","unpaid invoice"],["03:8","02:9"])
q(SF,"easy","How long are backups retained?","35 days.",["35 days"],["04:6"])
q(SF,"easy","What is the default and maximum page size on list endpoints?","Default 25, maximum 100.",["default 25","maximum 100"],["03:5"])
q(SF,"easy","What user roles exist in a Northwind organization?","Owner, Admin, Developer, Billing and Viewer.",["Owner","Admin","Developer","Billing","Viewer"],["05:7","04:4"])
q(SF,"easy","When is the scheduled maintenance window?","Sundays between 02:00 and 06:00 UTC.",["Sundays","02:00-06:00 UTC"],["08:6"])
q(SF,"easy","Who is allowed to cancel a subscription?","Only users with the Owner role.",["Owner role only"],["02:8"])
q(SF,"medium","What is the first-response target for an Enterprise Sev1 ticket?","15 minutes, 24/7.",["15 minutes","24/7"],["08:9"])
q(SF,"easy","Which minimum Python version does the Northwind Python SDK require?","Python 3.9.",["Python 3.9"],["03:10","06:13"])
q(SF,"easy","How many active API keys can a Starter project have?","10.",["10 active keys"],["03:2","05:11"])
q(SF,"easy","How long does a password reset link remain valid?","60 minutes.",["60 minutes"],["06:9"])

# ---------------- NUMERIC / CALCULATION ----------------
NC = "numeric_calculation"
q(NC,"medium","How much money do I save per year by paying for Starter annually instead of monthly?","$58 (12 × $29 = $348 monthly vs $290 annual).",["$348","$290","$58"],["01:3"])
q(NC,"medium","A Growth customer used 5,400,000 requests in a billing cycle. What overage will they be charged for requests?","$20.00: 400,000 extra requests = 40 blocks of 10,000 at $0.50 each.",["400,000 over","40 blocks","$0.50 per block","$20"],["01:4","01:2"])
q(NC,"hard","A Starter customer used 612,500 requests in a billing cycle. What is the request overage charge?","$9.60: 112,500 extra requests rounds up to 12 blocks of 10,000, at $0.80 each.",["112,500 over","rounded up to 12 blocks","$0.80 per block","$9.60"],["01:4","01:2"])
q(NC,"medium","A Starter organization has 5 members. How much extra do they pay per month for seats?","$16 per month: 3 seats are included, so 2 extra seats at $8 each.",["3 included","2 extra","$8 per seat","$16"],["01:2","01:4"])
q(NC,"medium","A Growth customer stores 520 GB. What is the monthly storage overage?","$1.20: 20 GB over the 500 GB included at $0.06 per GB.",["500 GB included","20 GB over","$0.06/GB","$1.20"],["01:2","01:4"])
q(NC,"hard","A customer buys the Growth annual plan ($1,490) and cancels 45 days later. How much is refunded?","$1,241.67: two months count as used (a partial month counts as full), so 10 unused months × ($1,490 ÷ 12).",["within 90 days","2 months used","10 unused","$1,241.67"],["02:4"],notes="Tests rounding of partial month and proration formula.")
q(NC,"medium","How many minutes of downtime are allowed in a 30-day month for a 99.5% uptime commitment?","216 minutes (0.5% of 43,200 minutes), i.e. 3.6 hours.",["43,200 minutes","0.5%","216 minutes"],["08:2","08:1"],notes="Not in the reference table; must be computed.")
q(NC,"medium","A Growth customer paying $149/month had 98.7% uptime in a month. What service credit are they due?","25% of $149 = $37.25 (uptime below 99.0%).",["below 99.0%","25%","$37.25"],["08:3"])
q(NC,"medium","A Starter customer paying $29/month had 99.3% uptime. What credit do they get?","5% of $29 = $1.45 (below 99.5% but at least 99.0%).",["99.3% is between 99.0 and 99.5","5%","$1.45"],["08:3"])
q(NC,"medium","By what percentage was the Growth plan rate limit increased in v3.8?","25% (from 2,400 to 3,000 requests per minute).",["2,400","3,000","25%"],["07:1"])
q(NC,"easy","I send one batch call containing 80 documents. How many requests does that count as?","80 requests; each document in a batch counts as one request.",["80 requests"],["03:3"])
q(NC,"hard","A company qualifies for the startup program on the Growth plan billed monthly. How much do they pay in total over the first 6 months?","$447: 50% off $149 = $74.50 per month × 6.",["50% off","$74.50","6 months","$447"],["01:6","01:2"])
q(NC,"medium","A nonprofit buys the Growth annual plan. What do they pay for the year?","$1,117.50: 25% nonprofit discount applied to $1,490.",["25% nonprofit discount","$1,490","$1,117.50"],["01:6","01:3"])
q(NC,"easy","How many times higher is the Starter rate limit than the Free rate limit?","10 times (600 vs 60 requests per minute).",["600","60","10x"],["03:3"])

# ---------------- VERSION CONFLICT ----------------
VC = "version_conflict"
q(VC,"medium","What is the current monthly price of the Starter plan?","$29 per month (raised from $19 on 2026-01-01). The $19 figure in the archived 2025 document is outdated.",["$29","effective 2026-01-01","$19 is outdated"],["01:2","07:8"],["09"])
q(VC,"medium","What is the refund (money-back) window for new paid customers?","14 days. It was extended from 7 days on 2026-01-01; the 7-day figure is outdated.",["14 days","was 7 days"],["02:2","07:8"],["09"])
q(VC,"medium","What is the current API rate limit for the Growth plan?","3,000 requests per minute, raised from 2,400 on 2026-10-14.",["3,000","raised from 2,400"],["03:3","07:1"],["09"])
q(VC,"medium","What uptime SLA does the Enterprise plan have?","99.99%, raised from 99.95% on 2026-01-01.",["99.99%","was 99.95%"],["08:1","07:8"],["09"])
q(VC,"medium","How much storage does the Free plan include?","1 GB (increased from 500 MB on 2026-01-01).",["1 GB","was 500 MB"],["01:2","07:8"],["09"])
q(VC,"easy","Can I still use API v2?","No. API v2 was shut down on 2026-03-31 and now returns 410 Gone; use v3.",["shut down 2026-03-31","410 Gone","use v3"],["03:11","07:5","05:10"])
q(VC,"medium","How many times will Northwind retry a failed webhook delivery?","Up to 8 attempts (increased from 5 on 2026-03-02), with exponential backoff over 24 hours.",["8 attempts","was 5"],["03:9","07:6"])
q(VC,"medium","How long are audit logs retained on the Growth plan?","30 days (extended from 14 days on 2026-10-14).",["30 days","was 14 days"],["04:5","07:1"])
q(VC,"medium","How long is an idempotency key remembered?","24 hours (extended from 12 hours in v3.7, 2026-08-05).",["24 hours","was 12 hours"],["03:6","07:2"])
q(VC,"medium","How many requests per month does the Starter plan include?","500,000 (increased from 250,000 on 2026-01-01).",["500,000","was 250,000"],["01:2","07:8"],["09"])
q(VC,"hard","How old can a webhook timestamp be before I should reject it, and has that changed?","More than 5 minutes old should be rejected. The tolerance was reduced from 10 minutes to 5 minutes on 2026-03-02.",["5 minutes","was 10 minutes"],["03:9","07:6"])

# ---------------- MULTI-HOP ----------------
MH = "multi_hop"
q(MH,"hard","I'm on the Starter plan and want SAML single sign-on. What do I need to do and what would it cost?","SAML SSO is available on Enterprise only, so you must move to an Enterprise plan, which has custom pricing with a minimum of $1,200 per month.",["SAML = Enterprise only","$1,200/month minimum"],["04:4","01:2"])
q(MH,"hard","My Free account returns 429 quota_exceeded. When does it reset, and what is the cheapest way to get more requests?","The Free quota resets on the 1st of the month at 00:00 UTC. The cheapest upgrade is Starter at $29/month, which includes 500,000 requests.",["resets 1st 00:00 UTC","Starter $29","500,000 requests"],["03:4","06:2","01:2"])
q(MH,"hard","I cancel my Starter plan and I have 20 GB of data. What happens to it?","The account reverts to Free at the end of the paid period. Because 20 GB exceeds the 1 GB Free limit, the project becomes read-only for 30 days and is then permanently deleted, so export first.",["reverts to Free","exceeds 1 GB","read-only 30 days","then deleted"],["02:8","01:2"])
q(MH,"hard","If my payment fails, when does the API stop working and when is my data deleted?","Retries occur on days 1, 3 and 5; if still unpaid on day 7 the account is suspended and the API returns 402. Data is deleted 30 days after the suspension begins unless paid.",["retries day 1,3,5","suspended day 7","402","30 days after suspension"],["02:9","03:8"])
q(MH,"hard","A Growth customer paying $149/month had a single 3-hour outage in a 30-day month and no other downtime. Do they get a credit, how much, and how do they claim it?","3 hours = 180 minutes of 43,200, so uptime is about 99.58%, below the 99.9% commitment but above 99.0%, giving a 10% credit ($14.90). It is not automatic: they must open a support ticket within 30 days after month end; the credit is applied to a future invoice, not paid in cash.",["99.58%","10%","$14.90","claim within 30 days via ticket","not cash"],["08:2","08:3","08:4"])
q(MH,"medium","Which plan do I need if I require both a HIPAA BAA and 24/7 phone support?","Enterprise. The HIPAA BAA is Enterprise only and 24/7 phone support is part of Enterprise.",["Enterprise"],["04:12","08:7"])
q(MH,"hard","I need EU data residency and audit logs retained for a full year. Which plan do I need?","Enterprise. EU data residency (eu-central-1) is available on all plans, but only Enterprise retains audit logs for 365 days.",["eu-central-1 on all plans","365-day audit logs = Enterprise"],["04:3","04:5"])
q(MH,"medium","A webhook delivery arrives with a timestamp 7 minutes old. Should my handler accept it?","No. Handlers should reject timestamps older than 5 minutes.",["reject","5 minutes"],["03:9","07:6"])
q(MH,"hard","I want to rotate an API key without downtime on the Growth plan. How, and how many keys can I have?","Use grace rotation, which keeps the old key valid for up to 24 hours. Growth projects can have up to 50 active keys.",["grace rotation 24 hours","50 keys"],["03:2","05:11"])
q(MH,"medium","I upgrade from Starter to Growth in the middle of my billing period. How am I charged and when do I get the higher rate limit?","You are charged a prorated amount for the rest of the billing period and the upgrade (including the 3,000/min rate limit and higher quota) takes effect immediately.",["prorated","immediate","3,000 requests/min"],["02:6","01:8","03:3"])
q(MH,"hard","I am on the Growth plan with a Sev1 incident. What is my first-response target, and how could I improve it?","2 hours, 24/7. Adding the Premier Support add-on ($499/month) improves it to 30 minutes, 24/7 and adds phone support for Sev1.",["2 hours","Premier Support $499","30 minutes"],["08:9","08:12"])
q(MH,"hard","My Python code gets 413 errors uploading 30 MB files on Starter. What are my options?","The Starter payload limit is 10 MB. Options: compress or split the file, or upgrade to Growth (50 MB limit, $149/month).",["Starter limit 10 MB","Growth 50 MB","compress/split"],["06:4","03:5","01:2"])

# ---------------- TRICKY / AMBIGUOUS ----------------
TA = "tricky_ambiguous"
q(TA,"medium","What is the rate limit?","It depends on the plan: Free 60/min, Starter 600/min, Growth 3,000/min, Enterprise 10,000/min by default (up to 100,000). Sandbox (nw_test_) keys are 100/min on any plan. A good answer lists these or asks which plan.",["Free 60","Starter 600","Growth 3,000","Enterprise 10,000","sandbox 100"],["03:3","01:2"],["09"],notes="Underspecified question; tests whether the system surfaces the plan dependence.")
q(TA,"medium","Can I get a refund?","It depends: within 14 days of the first paid charge, a full refund; monthly plans are otherwise non-refundable; annual plans get a prorated refund if cancelled within 90 days; overage charges are never refundable.",["14-day full refund","monthly no refund after","annual prorated within 90 days","overage non-refundable"],["02:2","02:3","02:4","02:5"],notes="Underspecified question.")
q(TA,"medium","Is the rate limit applied per API key?","No. It is applied per project, so multiple keys in the same project share one limit.",["per project","not per key"],["03:3","06:2"])
q(TA,"easy","Does the Free plan come with an uptime SLA or service credits?","No. The Free plan has no uptime commitment (best effort) and is not eligible for service credits.",["no SLA","not eligible for credits"],["08:1","08:3"])
q(TA,"medium","If I downgrade from Growth to Starter, do I get money back for the rest of the month?","No. Downgrades take effect at the end of the current billing period and no credit or refund is given for the unused part of the higher plan.",["end of period","no refund"],["02:6"])
q(TA,"medium","Can I get my overage charges refunded under the 14-day money-back guarantee?","No. Overage charges are never refundable, even within the 14-day window.",["overage never refundable"],["02:5","02:2"])
q(TA,"medium","Does scheduled maintenance count as downtime for SLA credits?","No, scheduled maintenance does not count (Sundays 02:00-06:00 UTC with 7 days' notice). Emergency maintenance does count.",["scheduled excluded","emergency counts"],["08:5","08:6"])
q(TA,"hard","Does my 14-day free trial count towards the 14-day money-back guarantee?","No. The money-back window starts at the first paid charge, not at the start of the trial.",["window starts at first paid charge"],["02:2","01:5"])
q(TA,"medium","Is SMS a supported MFA method?","No. Supported MFA methods are authenticator apps (TOTP) and WebAuthn security keys; SMS is not supported.",["TOTP","WebAuthn","no SMS"],["04:4"])
q(TA,"medium","Is a 4-hour Sev2 response guaranteed for Growth customers?","Not by default: the Growth Sev2 first-response target is 8 business hours. 4 hours (24/7) applies only with the Premier Support add-on. Also, targets are for first response, not resolution.",["8 business hours default","4 hours only with Premier","first response not resolution"],["08:9","08:12"])

# ---------------- PROCEDURAL ----------------
PR = "procedural"
q(PR,"easy","How do I cancel my subscription?","An Owner goes to Settings > Billing > Cancel plan. Paid features remain until the end of the paid period, then the account reverts to Free.",["Owner","Settings > Billing > Cancel plan"],["02:8"])
q(PR,"hard","How do I verify a webhook signature?","Read X-Northwind-Signature (t=<timestamp>,v1=<signature>). Compute HMAC-SHA256 of '<timestamp>.<raw request body>' with your endpoint secret, compare it with v1, and reject if the timestamp is older than 5 minutes.",["HMAC-SHA256","timestamp.raw body","endpoint secret","reject older than 5 min"],["03:9","06:7"])
q(PR,"easy","How do I export my data?","Go to Settings > Data export and request an export. A ZIP is prepared within 24 hours and the download link is valid for 7 days.",["Settings > Data export","24 hours","link valid 7 days"],["06:11"])
q(PR,"easy","How do I claim an SLA service credit?","Open a support ticket within 30 days after the end of the month of the downtime, with the affected dates and times.",["support ticket","within 30 days of month end"],["08:4"])
q(PR,"medium","How should my client handle 429 rate_limited responses?","Honor the Retry-After header and retry with exponential backoff and jitter (start 500 ms, cap 30 s). Remember the limit is per project.",["Retry-After","exponential backoff with jitter"],["03:3","06:2"])
q(PR,"medium","How do I move my project to a different region?","The region cannot be changed. Create a new project in the target region and use the export/import tool.",["cannot change region","new project","export/import"],["04:3"])
q(PR,"easy","How do I report a security vulnerability?","Email security@northwindcloud.example; reports are acknowledged within 2 business days and bounties go up to $10,000 for critical issues.",["security@northwindcloud.example","2 business days"],["04:9"])
q(PR,"medium","I lost my MFA device and have no recovery codes. What can I do?","Another Owner or Admin can reset your MFA. If you are the sole Owner, contact support and complete identity verification, which can take up to 3 business days.",["Owner/Admin can reset","sole owner contacts support","up to 3 business days"],["06:9"])
q(PR,"medium","What do I need to change to migrate from API v2 to v3?","Change /v2 to /v3, replace the X-API-Key header with Authorization: Bearer, switch from page to cursor pagination, and handle v3's string error codes.",["/v3","Bearer","cursor pagination","string error code"],["05:10"])

# ---------------- UNANSWERABLE ----------------
UA = "unanswerable"
q(UA,"easy","Does Northwind offer an official Ruby SDK?","Not stated in the documentation. The documented SDKs are Python, Node.js and Go.",["not in docs"],["03:10"],answerable=False,notes="Correct behavior: say it is not documented; may mention the three documented SDKs.")
q(UA,"easy","Is Northwind Cloud FedRAMP authorized?","Not stated in the documentation. The docs list SOC 2 Type II, ISO 27001, GDPR/CCPA, and HIPAA/PCI notes only.",["not in docs"],["04:1","04:12"],answerable=False)
q(UA,"easy","What is the maximum number of documents I can store per project?","The documentation does not specify a per-project document count limit (only storage size and request limits).",["not in docs"],["01:2","03:5"],answerable=False)
q(UA,"easy","Who is the CEO of Northwind Cloud?","Not stated in the documentation.",["not in docs"],["01:1"],answerable=False)
q(UA,"medium","How much would an Enterprise plan cost for 500 seats?","Not stated. Enterprise pricing is custom with a minimum of $1,200 per month; seat-specific pricing is not published.",["not in docs","custom pricing"],["01:2"],answerable=False)
q(UA,"easy","Will the Starter plan price change in 2027?","Not stated. The docs only say price increases are announced at least 60 days in advance.",["not in docs","60 days notice"],["07:10"],answerable=False)
q(UA,"easy","Does the Python SDK support asyncio?","Not stated in the documentation.",["not in docs"],["03:10","05:5"],answerable=False)
q(UA,"medium","How many minutes of downtime did the API have last August?","Not stated. The docs contain no historical uptime data; see the status page for incidents.",["not in docs"],["08:2","08:11"],answerable=False)
q(UA,"medium","What were the rate limits for the old v2 API?","Not stated in the current documentation. v2 was shut down on 2026-03-31 and its limits are not documented.",["not in docs"],["03:11","07:5"],["09"],answerable=False,notes="The archived 2025 pricing doc has plan limits but does not say they were v2 limits; a system that asserts they were is hallucinating.")
q(UA,"easy","Can I pay for my subscription with PayPal?","Not stated. The docs list Visa, Mastercard, American Express and, for annual Growth/Enterprise, ACH or wire invoice; PayPal is not mentioned.",["not in docs","documented methods only"],["01:7"],answerable=False)

if __name__ == "__main__":
    # validate sections exist
    errors = 0
    for e in EVALS:
        for ref in e["source_sections"]:
            fn, sec = ref.split("#")
            text = open(os.path.join(DOCS_DIR, fn), encoding="utf-8").read()
            if not re.search(rf"^## {sec}\. ", text, re.M):
                print("MISSING SECTION", e["id"], ref); errors += 1
    ids = [e["id"] for e in EVALS]
    assert len(ids) == len(set(ids))
    out_json = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    out_jsonl = os.path.join(os.path.dirname(__file__), "eval_dataset.jsonl")
    meta = {"dataset": "northwind-cloud-rag-evals", "version": "1.0", "as_of": "2026-10-14",
            "num_questions": len(EVALS), "docs_dir": "docs/"}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "questions": EVALS}, f, indent=2, ensure_ascii=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in EVALS: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    print("total", len(EVALS), "errors", errors)
    print(Counter(e["category"] for e in EVALS))
    print(Counter(e["difficulty"] for e in EVALS))
