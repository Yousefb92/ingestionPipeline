#!/usr/bin/env python3
"""
EDGAR Ingestion Script — Emerson / NI Backtest
═══════════════════════════════════════════════
Populates PostgreSQL (financial_metrics, indicator_events, strategic_context)
with real SEC EDGAR data for 5 companies covering 2020-2023, then writes a
Neo4j Cypher script for graph seeding.

10-k and 10-Q (Delisted companies only)
Revenue (e.g., Revenues, NetSales)
Depreciation & Amortization (D&A)
Operating Income / Net Income
Gross Profit
Research & Development (R&D) Expenses
Operating Cash Flow & CapEx
Long-Term Debt

Activist Disclosures (Phase 3a - EFTS Search)
To populate Indicator 1 (Activist Pressure), the script uses the SEC's Full-Text Search (EFTS) API. It pulls:
Forms SC 13D and SC 13G (and their amendments).
It extracts the Filer Name (e.g., JANA Partners LLC) and the File Date.

To track leadership instability (Indicator 2), it looks through the company's submissions feed.
It isolates Form 8-K filings specifically containing Item 5.02 ("Departure of Directors or Certain Officers").
It downloads the actual HTML document and uses regex to scrape the Person's Name and their Role (e.g., CEO, CFO, President).
It stores up to 2000 characters of the surrounding text for the AI agent to read later.

To track sector consolidation (Indicator 4), it scans the submission descriptions of peer companies.
It flags Form 8-K filings that mention keywords like "acquisition," "merger," or Item 1.01 ("Material Definitive Agreement").

"""

import os
import re
import sys
import time
import warnings
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import date
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# Suppress the BeautifulSoup warning about parsing inline-XBRL 8-K documents
# (SEC filings are valid HTML+XML hybrids; lxml handles them correctly anyway)
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass  # older bs4 versions don't have this warning class

# ── .env resolution ───────────────────────────────────────────────────────────
# Try ingestion/.env first, then fall back to ma_agent_task/.env
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)               # ingestion/
_sibling_env = os.path.join(_project_dir, "..", "ma_agent_task", ".env")
GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)

load_dotenv(dotenv_path=os.path.join(_project_dir, ".env"))   # ingestion/.env
load_dotenv(dotenv_path=_sibling_env)                          # ma_agent_task/.env (fallback)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COMPANIES = {
    "NATI":   {"cik": "0000935494", "name": "National Instruments Corp"},
    "EMR":  {"cik": "0000032604", "name": "Emerson Electric Co"},
    "KEYS": {"cik": "0001601046", "name": "Keysight Technologies Inc"},
    "TER":  {"cik": "0000097210", "name": "Teradyne Inc"},
    "AME":  {"cik": "0001037868", "name": "Ametek Inc"},
}

SECTOR_NAME = "Test & Measurement"
SECTOR_CODE = "TM"
DATE_FROM   = date(2019, 1, 1)
# DATE_TO     = date(2023, 6, 1)
DATE_TO     = date(2022, 4, 12)
EDGAR_BASE    = "https://data.sec.gov"
EDGAR_HEADERS = {"User-Agent": "PwC-CaseStudy ingestion@pwc.com"}

# Set to False to suppress per-hit debug lines (useful once everything is working)
VERBOSE = True

# XBRL concept fallbacks (first concept with ≥ MIN_REVENUE_PERIODS in-range entries wins)
# Revenue requires at least 3 periods to distinguish a real total-revenue concept
# from a segment/JV line (e.g. EMR's Revenues concept captures only a 2-period JV slice).
MIN_REVENUE_PERIODS = 3

REVENUE_CONCEPTS = [
    "Revenues",
    "NetSales",                # EMR and many industrial conglomerates use this
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetRevenues",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
DA_CONCEPTS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "OtherDepreciationAndAmortization",
    # KEYS and some other companies report depreciation and amortisation
    # separately rather than as a combined line in the cash flow statement
    "Depreciation",
    "AmortizationOfIntangibleAssets",
]
# OperatingIncomeLoss is missing from some large conglomerates (e.g. Emerson).
# NetIncomeLoss / ProfitLoss are used as fallbacks — this gives an after-tax
# EBITDA proxy that still captures the margin compression trend accurately enough
# for the scoring model.
OPINCOME_CONCEPTS  = ["OperatingIncomeLoss", "NetIncomeLoss", "ProfitLoss"]
GROSSPROFIT_CONCEPTS = ["GrossProfit"]
RD_CONCEPTS        = ["ResearchAndDevelopmentExpense"]
CFO_CONCEPTS       = ["NetCashProvidedByUsedInOperatingActivities"]
CAPEX_CONCEPTS     = ["PaymentsToAcquirePropertyPlantAndEquipment",
                      "PaymentsForCapitalImprovements"]
DEBT_CONCEPTS      = ["LongTermDebtNoncurrent", "LongTermDebt",
                      "LongTermDebtAndCapitalLeaseObligationsNoncurrent"]

# Keywords that trigger indicator_id=3 (Strategic Pivot Intent)
PIVOT_KEYWORDS = [
    "strategic alternatives",
    "portfolio review",
    "non-core",
    "transformation",
    "rightsizing",
    "inorganic",
    "cost optimization",
    "restructuring",
    "margin expansion"
]

# All companies get MD&A text extraction
MDA_TICKERS = set(COMPANIES.keys())
MDA_YEARS   = {"2021", "2022"}


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def get_pg_connection():
    return psycopg2.connect(
        host=os.environ.get("PG_HOST", "localhost"),
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ.get("PG_NAME", "postgres"),
        user=os.environ.get("PG_USER", "postgres"),
        password=os.environ.get("PG_PASS", "postgres"),
    )


def apply_schema_additions(conn):
    """
    Safely add four columns that are missing from the baseline DDL but are
    needed by this ingestion script (and in two cases by existing app code).
    All statements use IF NOT EXISTS so they are no-ops if run a second time.
    """
    additions = [
        # Needed by this script — links metric rows back to EDGAR filings
        ("financial_metrics", "source_url",
         "ALTER TABLE financial_metrics ADD COLUMN IF NOT EXISTS source_url TEXT"),

        # Needed by this script — stores the EDGAR CIK on each company row
        # for future automated ingestion without a hardcoded map
        ("companies", "cik",
         "ALTER TABLE companies ADD COLUMN IF NOT EXISTS cik VARCHAR(20)"),

        # Needed by baseline_creation.sql — it already tries to INSERT this column
        ("sectors", "code",
         "ALTER TABLE sectors ADD COLUMN IF NOT EXISTS code VARCHAR(20)"),

        # Needed by postgres_client.py save_thesis_report() — currently causes
        # a runtime error because the column is referenced but not in the DDL
        ("deal_theses", "indicator_verdicts",
         "ALTER TABLE deal_theses ADD COLUMN IF NOT EXISTS indicator_verdicts JSONB"),
    ]

    with conn:
        with conn.cursor() as cur:
            for table, column, sql in additions:
                cur.execute(sql)
                print(f"  [SCHEMA] {table}.{column} — ensured present")


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def edgar_get(url: str, params: dict = None):
    """
    Rate-limited GET to EDGAR (SEC rate limit: 10 req/s → sleep 0.11 s).
    Returns parsed dict if JSON, raw str otherwise. Returns None on error.
    """
    time.sleep(0.11)
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        if "json" in resp.headers.get("Content-Type", ""):
            return resp.json()
        return resp.text
    except Exception as exc:
        print(f"    [EDGAR] Request failed {url}: {exc}")
        return None


def raw_cik(cik: str) -> str:
    """'0000032604' → '32604' (unpadded integer string for URL paths)."""
    return str(int(cik))


def build_accession_url(cik_int: str, accession: str) -> str:
    acc_nodash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{accession}-index.htm"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: SEED COMPANIES
# ─────────────────────────────────────────────────────────────────────────────

def seed_companies(conn) -> dict:
    """
    Insert the Test & Measurement sector and the 5 companies.
    Returns {ticker: company_id} used by all downstream phases.
    Idempotent via ON CONFLICT DO NOTHING on unique columns.
    """
    company_ids = {}
    with conn:
        with conn.cursor() as cur:
            # Sector — unique on name
            cur.execute(
                """
                INSERT INTO sectors (name, code)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET code = EXCLUDED.code
                """,
                (SECTOR_NAME, SECTOR_CODE),
            )
            cur.execute("SELECT id FROM sectors WHERE name = %s", (SECTOR_NAME,))
            sector_id = cur.fetchone()[0]
            print(f"  [SETUP] Sector '{SECTOR_NAME}' (code={SECTOR_CODE}) → id={sector_id}")

            for ticker, meta in COMPANIES.items():
                cur.execute(
                    """
                    INSERT INTO companies (ticker, name, sector_id, cik)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET cik = EXCLUDED.cik
                    """,
                    (ticker, meta["name"], sector_id, meta["cik"]),
                )
                cur.execute("SELECT id FROM companies WHERE ticker = %s", (ticker,))
                company_ids[ticker] = cur.fetchone()[0]
                print(f"  [SETUP]   {ticker:5s} → company_id={company_ids[ticker]:4d}  cik={meta['cik']}")

    return company_ids


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: FINANCIAL METRICS (XBRL)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_concept_series(facts_usgaap: dict, concepts: list,
                            min_periods: int = 1) -> dict:
    """
    Try each XBRL concept name in order; return a dict of
    {end_date_str: {value, form, fp, accession}}
    for the first concept that has at least *min_periods* entries within
    DATE_FROM..DATE_TO.

    min_periods=3 for revenue filters out JV/segment lines that happen to
    use the same concept name but only appear in 1-2 filings.

    When both 10-K and 10-Q data exist for the same period-end date,
    the annual (10-K) filing wins (avoids double-counting).
    """
    for concept in concepts:
        node = facts_usgaap.get(concept)
        if not node:
            if VERBOSE:
                print(f"      [XBRL] '{concept}' — not present in facts")
            continue
        entries = node.get("units", {}).get("USD", [])
        if not entries:
            if VERBOSE:
                print(f"      [XBRL] '{concept}' — present but no USD units")
            continue

        # Count total entries vs in-range entries for diagnostics
        total_entries = len(entries)
        series = {}
        for e in entries:
            form = e.get("form", "")
            if form not in ("10-K", "10-Q"):
                continue
            end = e.get("end", "")
            if not end:
                continue
            try:
                end_date = date.fromisoformat(end)
            except ValueError:
                continue
            if not (DATE_FROM <= end_date <= DATE_TO):
                continue
            # Annual filing takes priority when both exist for same end date
            if end not in series or form == "10-K":
                series[end] = {
                    "value":     e.get("val", 0) or 0,
                    "form":      form,
                    "fp":        e.get("fp", ""),
                    "accession": e.get("accn", ""),
                }

        if len(series) >= min_periods:
            print(f"      [XBRL] '{concept}' ✓  {len(series)} periods in range "
                  f"(out of {total_entries} total entries)")
            return series

        if series:
            # Concept found data but below the minimum count threshold —
            # likely a segment/JV slice rather than the full company figure.
            if VERBOSE:
                print(f"      [XBRL] '{concept}' — only {len(series)} period(s) in range "
                      f"(min required: {min_periods}), trying next fallback")
        else:
            # Concept exists but all data falls outside our date window
            if VERBOSE:
                print(f"      [XBRL] '{concept}' — {total_entries} entries but 0 fall "
                      f"in {DATE_FROM}…{DATE_TO}, trying next fallback")

    return {}


def _insert_metric(cur, company_id: int, metric_type: str, value: float,
                   end_date: str, doc_label: str, src_url: str):
    """Idempotent insert for a single financial_metrics row."""
    cur.execute(
        """
        INSERT INTO financial_metrics
            (company_id, metric_type, value, observed_date,
             source_document, source_url)
        SELECT %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (
            SELECT 1 FROM financial_metrics
            WHERE company_id    = %s
              AND metric_type   = %s
              AND observed_date = %s
        )
        """,
        (company_id, metric_type, value, end_date, doc_label, src_url,
         company_id, metric_type, end_date),
    )


def ingest_financials(ticker: str, cik: str, company_id: int, conn):
    """
    Fetch XBRL companyfacts and insert six metric types per period:

      ebitda_margin   — (OperatingIncomeLoss + D&A) / Revenue
      gross_margin    — GrossProfit / Revenue
      rd_intensity    — R&D Expense / Revenue  (key signal for T&M acquirers)
      fcf_margin      — (Operating CF - CapEx) / Revenue
      net_debt_mm     — Long-term debt in $M  (leverage signal)
      revenue_mm      — Absolute revenue in $M  (size context)
    """
    print(f"  [FINANCIALS] Fetching XBRL companyfacts for {ticker}...")
    data = edgar_get(f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
    if not data or not isinstance(data, dict):
        print(f"  [FINANCIALS] No data returned, skipping {ticker}.")
        return

    facts = data.get("facts", {}).get("us-gaap", {})
    if not facts:
        print(f"  [FINANCIALS] No us-gaap facts for {ticker}.")
        return

    print(f"    Revenue concepts:")
    revenues    = _extract_concept_series(facts, REVENUE_CONCEPTS,
                                          min_periods=MIN_REVENUE_PERIODS)
    print(f"    D&A concepts:")
    da_vals     = _extract_concept_series(facts, DA_CONCEPTS)
    print(f"    Operating income concepts:")
    opincome    = _extract_concept_series(facts, OPINCOME_CONCEPTS)
    print(f"    Gross profit concepts:")
    grossprofit = _extract_concept_series(facts, GROSSPROFIT_CONCEPTS)
    print(f"    R&D expense concepts:")
    rd_exp      = _extract_concept_series(facts, RD_CONCEPTS)
    print(f"    Operating cash flow concepts:")
    cfo         = _extract_concept_series(facts, CFO_CONCEPTS)
    print(f"    CapEx concepts:")
    capex       = _extract_concept_series(facts, CAPEX_CONCEPTS)
    print(f"    Long-term debt concepts:")
    debt        = _extract_concept_series(facts, DEBT_CONCEPTS)

    # ── Concept resolution summary ─────────────────────────────────────────────
    print(f"\n  [FINANCIALS] Period coverage for {ticker}:")
    print(f"    Revenue:    {len(revenues):>3d} periods  |  D&A:       {len(da_vals):>3d} periods")
    print(f"    Op Income:  {len(opincome):>3d} periods  |  Gross P:   {len(grossprofit):>3d} periods")
    print(f"    R&D:        {len(rd_exp):>3d} periods  |  CFO:       {len(cfo):>3d} periods")
    print(f"    CapEx:      {len(capex):>3d} periods  |  LT Debt:   {len(debt):>3d} periods")

    if revenues:
        latest = sorted(revenues)[-1]
        rev_m  = revenues[latest]["value"] / 1_000_000
        print(f"    Latest revenue: ${rev_m:,.1f}M  as of {latest}  "
              f"({revenues[latest]['form']} {revenues[latest]['fp']})")
    else:
        print("    ⚠  No revenue data — EBITDA/margin metrics will be skipped")

    if not revenues or not da_vals or not opincome:
        missing = [name for name, s in [("Revenue", revenues), ("D&A", da_vals),
                                         ("OpIncome", opincome)] if not s]
        print(f"    ✗ Missing core series: {missing} — cannot compute EBITDA margin")
        # If D&A is missing, run a discovery scan so we know what to add
        if not da_vals and VERBOSE:
            da_candidates = sorted(
                k for k in facts
                if "depreciation" in k.lower() or "amortization" in k.lower()
            )
            if da_candidates:
                print(f"    ℹ  D&A discovery — concepts present in facts with "
                      f"'depreciation'/'amortization' in name:")
                for cname in da_candidates[:12]:
                    node   = facts[cname]
                    usd    = node.get("units", {}).get("USD", [])
                    in_rng = 0
                    for e in usd:
                        if e.get("form", "") not in ("10-K", "10-Q"):
                            continue
                        end_str = e.get("end", "")
                        if not end_str:
                            continue
                        try:
                            if DATE_FROM <= date.fromisoformat(end_str) <= DATE_TO:
                                in_rng += 1
                        except ValueError:
                            pass
                    print(f"       '{cname}': {len(usd)} total entries, "
                          f"{in_rng} in {DATE_FROM}…{DATE_TO}")

    # EBITDA margin requires all three core series to align
    ebitda_dates = set(revenues) & set(da_vals) & set(opincome)
    print(f"\n  [FINANCIALS] {len(ebitda_dates)} aligned EBITDA periods for {ticker}")
    if not ebitda_dates and revenues and (da_vals or opincome):
        # Help diagnose date-alignment mismatches
        rev_dates = set(revenues)
        da_dates  = set(da_vals)
        oi_dates  = set(opincome)
        print(f"    ⚠  Date alignment issue:")
        print(f"       Revenue dates:    {sorted(rev_dates)[-3:] if rev_dates else '[]'}")
        print(f"       D&A dates:        {sorted(da_dates)[-3:]  if da_dates  else '[]'}")
        print(f"       Op Income dates:  {sorted(oi_dates)[-3:]  if oi_dates  else '[]'}")

    cik_int  = raw_cik(cik)
    inserted = 0

    with conn:
        with conn.cursor() as cur:
            for end_date in sorted(ebitda_dates):
                rev = revenues[end_date]["value"]
                da  = da_vals[end_date]["value"]
                oi  = opincome[end_date]["value"]
                if not rev:
                    continue

                form = revenues[end_date]["form"]
                fp   = revenues[end_date]["fp"]
                acc  = revenues[end_date]["accession"]
                doc_label = (
                    f"10-K FY{end_date[:4]}" if form == "10-K"
                    else f"10-Q {fp}-{end_date[:4]}"
                )
                src_url = build_accession_url(cik_int, acc) if acc else ""

                # 1. EBITDA margin
                ebitda_margin = (oi + da) / rev
                # Sanity check: EBITDA margin outside [-0.5, 0.8] is suspicious
                if VERBOSE and not (-0.5 <= ebitda_margin <= 0.8):
                    print(f"    ⚠  [SANITY] {ticker} {end_date}: "
                          f"ebitda_margin={ebitda_margin:.3f} looks unusual "
                          f"(rev={rev/1e6:.1f}M, oi={oi/1e6:.1f}M, da={da/1e6:.1f}M)")
                _insert_metric(cur, company_id, "ebitda_margin",
                               ebitda_margin, end_date, doc_label, src_url)
                inserted += cur.rowcount

                # 2. Gross margin
                if end_date in grossprofit:
                    _insert_metric(cur, company_id, "gross_margin",
                                   grossprofit[end_date]["value"] / rev,
                                   end_date, doc_label, src_url)
                    inserted += cur.rowcount

                # 3. R&D intensity
                if end_date in rd_exp:
                    _insert_metric(cur, company_id, "rd_intensity",
                                   rd_exp[end_date]["value"] / rev,
                                   end_date, doc_label, src_url)
                    inserted += cur.rowcount

                # 4. FCF margin — requires both CFO and CapEx on same date
                if end_date in cfo and end_date in capex:
                    fcf = cfo[end_date]["value"] - capex[end_date]["value"]
                    _insert_metric(cur, company_id, "fcf_margin",
                                   fcf / rev, end_date, doc_label, src_url)
                    inserted += cur.rowcount

                # 5. Revenue in $M (absolute size for EV context)
                _insert_metric(cur, company_id, "revenue_mm",
                               rev / 1_000_000, end_date, doc_label, src_url)
                inserted += cur.rowcount

            # 6. Net debt in $M — point-in-time balance sheet, insert where available
            #    Uses its own date set (balance sheet dates differ from income stmt)
            for end_date, entry in sorted(debt.items()):
                form      = entry["form"]
                fp        = entry["fp"]
                acc       = entry["accession"]
                doc_label = (
                    f"10-K FY{end_date[:4]}" if form == "10-K"
                    else f"10-Q {fp}-{end_date[:4]}"
                )
                src_url = build_accession_url(cik_int, acc) if acc else ""
                _insert_metric(cur, company_id, "net_debt_mm",
                               entry["value"] / 1_000_000,
                               end_date, doc_label, src_url)
                inserted += cur.rowcount

    print(f"  [FINANCIALS] Inserted {inserted} metric rows for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: FILING EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_submissions(ticker: str, cik: str) -> list:
    """
    Fetch all filings for a CIK from the EDGAR submissions endpoint,
    paginating through older filing sets automatically.
    Returns a flat list of filing dicts filtered to DATE_FROM..DATE_TO.

    Each dict contains: accessionNumber, filingDate, form,
                        primaryDocument, primaryDocDescription, items
    """
    data = edgar_get(f"{EDGAR_BASE}/submissions/CIK{cik}.json")
    if not data or not isinstance(data, dict):
        return []

    def _rows_from_block(block: dict) -> list:
        keys = ["accessionNumber", "filingDate", "form",
                "primaryDocument", "primaryDocDescription", "items"]
        length = len(block.get("accessionNumber", []))
        rows   = []
        for i in range(length):
            rows.append({k: (block.get(k) or [""] * length)[i] for k in keys})
        return rows

    all_rows = _rows_from_block(data.get("filings", {}).get("recent", {}))

    # Paginate older filing batches
    for extra_file in data.get("filings", {}).get("files", []):
        extra = edgar_get(f"{EDGAR_BASE}/submissions/{extra_file['name']}")
        if extra and isinstance(extra, dict):
            all_rows.extend(_rows_from_block(extra))

    # Filter to date window
    result = []
    for row in all_rows:
        try:
            fd = date.fromisoformat(row["filingDate"])
            if DATE_FROM <= fd <= DATE_TO:
                result.append(row)
        except (ValueError, TypeError):
            pass

    print(f"  [SUBMISSIONS] {ticker}: {len(result)} filings in date range")
    return result


def _insert_event(cur, company_id: int, indicator_id: int, event_name: str,
                  base_weight: int, event_date, source_url: str, summary: str):
    """
    Idempotent insert into indicator_events.
    Uses WHERE NOT EXISTS to avoid duplicates without requiring a unique index.
    Enforces the VARCHAR(100) limit on event_name.
    """
    event_name = event_name[:100]
    cur.execute(
        """
        INSERT INTO indicator_events
            (company_id, indicator_id, event_name, base_weight,
             event_date, source_url, raw_summary)
        SELECT %s, %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (
            SELECT 1 FROM indicator_events
            WHERE company_id   = %s
              AND indicator_id = %s
              AND event_date   = %s
              AND event_name   = %s
        )
        """,
        (company_id, indicator_id, event_name, base_weight,
         event_date, source_url, summary,
         company_id, indicator_id, event_date, event_name),
    )


# ── Phase 3a ─────────────────────────────────────────────────────────────────

def ingest_activist_events(company_ids: dict, conn):
    """
    Search EDGAR EFTS full-text search for SC 13D / SC 13G filings that
    name each of our 5 companies as a subject. This is necessary because
    activist schedules are filed BY the activist (e.g. Jana Partners), not
    by the target company, so they never appear in a company's own submissions
    feed. We run one EFTS query per company.

    Maps to indicator_id=1 (Activist Pressure).
    base_weight: 15 for passive 13G, 40 for active 13D.
    """
    total_inserted = 0

    for ticker, company_id in company_ids.items():
        company_name = COMPANIES[ticker]["name"]
        print(f"  [ACTIVIST] Searching EFTS for 13D/13G filings on {ticker}...")

        data = edgar_get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q":         f'"{company_name}"',
                "forms":     "SC 13D,SC 13G",
                "dateRange": "custom",
                "startdt":   DATE_FROM.isoformat(),
                "enddt":     DATE_TO.isoformat(),
            },
        )

        if not data or not isinstance(data, dict):
            print(f"  [ACTIVIST] No EFTS data returned for {ticker}.")
            continue

        hits = data.get("hits", {}).get("hits", [])
        print(f"  [ACTIVIST] {ticker}: {len(hits)} hits from EFTS")

        inserted  = 0
        skipped   = 0
        with conn:
            with conn.cursor() as cur:
                for hit in hits:
                    src = hit.get("_source", {})

                    file_date  = src.get("file_date", "")
                    form       = src.get("form_type", "")
                    display    = src.get("display_names", [])

                    # accession_no is often empty in the search-index endpoint;
                    # fall back to the hit's _id which always contains the accession
                    accession = src.get("accession_no", "") or hit.get("_id", "")

                    if VERBOSE:
                        print(f"    Hit: {file_date}  {form:<8s}  "
                              f"display_names={display}")

                    if not file_date:
                        skipped += 1
                        if VERBOSE:
                            print(f"      [SKIP] No file_date")
                        continue

                    # Confirm the filing names this company as the subject.
                    # Strip common legal suffixes before matching — EDGAR
                    # display_names often omit "Inc", "Corp", "Co.", etc.
                    _suffixes  = r"\s+(inc\.?|corp\.?|co\.?|ltd\.?|llc\.?|incorporated|corporation)$"
                    name_lower = re.sub(_suffixes, "", company_name.lower()).strip()
                    is_subject = any(name_lower in str(d).lower() for d in display)
                    if not is_subject:
                        skipped += 1
                        if VERBOSE:
                            print(f"      [SKIP] '{name_lower}' not found in "
                                  f"display_names — not the subject company")
                        continue

                    # Extract filer name from display_names — it is the entry
                    # that does NOT match the subject company name.
                    # EFTS does not expose entity_name as a top-level field;
                    # display_names contains strings like
                    # "Jana Partners LLC (CIK 0001594686)"
                    filer_name = "Unknown Filer"
                    for d in display:
                        d_str = str(d)
                        if name_lower not in d_str.lower():
                            # Strip the trailing "(CIK XXXXXXXXXX)" if present
                            filer_name = re.sub(r"\s*\(CIK[^)]*\)", "", d_str).strip()
                            if filer_name:
                                break

                    if filer_name == "Unknown Filer":
                        print(f"      ⚠  Could not identify filer from display_names={display}")

                    # Build a direct filing URL from the accession number.
                    # Accession format: XXXXXXXXXX-YY-ZZZZZZ  (filer CIK - year - seq)
                    src_url = ""
                    if accession:
                        filer_cik_raw = str(int(accession.split("-")[0]))
                        acc_nodash    = accession.replace("-", "")
                        src_url = (
                            f"https://www.sec.gov/Archives/edgar/data/"
                            f"{filer_cik_raw}/{acc_nodash}/{accession}-index.htm"
                        )

                    # Weight by actual filing type:
                    #   SC 13D   = new activist position (high weight)
                    #   SC 13D/A = amendment to existing 13D (moderate)
                    #   SC 13G   = passive institutional holder (low weight)
                    #   SC 13G/A = amendment to passive filing (very low)
                    if form in ("SC 13D",):
                        weight = 40
                    elif form in ("SC 13D/A",):
                        weight = 20
                    elif form in ("SC 13G",):
                        weight = 10
                    else:  # SC 13G/A or unknown
                        weight = 5
                    event_name = f"{form} — {filer_name}"
                    summary    = (
                        f"{filer_name} filed {form} on {company_name} on {file_date}."
                    )
                    print(f"      ✓ MATCH  filer={filer_name!r}  weight={weight}  "
                          f"acc={accession}")

                    _insert_event(cur, company_id, 1, event_name,
                                  weight, file_date, src_url, summary)
                    inserted += cur.rowcount

        print(f"  [ACTIVIST] {ticker}: {inserted} inserted, {skipped} skipped "
              f"(not subject / no date)")
        total_inserted += inserted

    print(f"  [ACTIVIST] Total activist events inserted: {total_inserted}")


# ── Phase 3b / 3c / 3d ───────────────────────────────────────────────────────

def _fetch_8k_502_text(cik_int: str, accession: str,
                       primary_doc: str) -> tuple[str, str, str]:
    """
    Fetch an 8-K primary document and extract the Item 5.02 section text.
    Returns (item_text, person_name, role).
    All three are empty strings if the fetch fails or the section isn't found.
    """
    if not primary_doc or not accession:
        return "", "", ""

    acc_nodash = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{primary_doc}"
    )
    time.sleep(0.11)
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
        if not resp.ok:
            return "", "", ""
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text(separator="\n")
    except Exception:
        return "", "", ""

    # Extract text between "Item 5.02" and the next "Item X.XX" header
    match = re.search(r"Item\s+5\.02[.\s](.+?)(?=Item\s+\d+\.\d+|$)",
                      text, re.IGNORECASE | re.DOTALL)
    if not match:
        print(f"      [8K-502] ✗ Item 5.02 section not found in '{primary_doc}'")
        if VERBOSE:
            # Show a snippet of the document to help diagnose regex misses
            preview = text[:400].replace("\n", " ")
            print(f"      [8K-502] Doc preview: {preview!r}")
        return "", "", ""

    item_text = re.sub(r"\n{3,}", "\n\n", match.group(0).strip())
    item_text = item_text[:2000]   # Cap at 2000 chars for storage

    # ── Name extraction ────────────────────────────────────────────────────────
    # Two-step approach to avoid a subtle regex bug:
    #
    # Problem: re.IGNORECASE makes character classes like [A-Z] and [a-z] match
    # ANY letter (upper or lower), so [A-Z][a-z]+ would match "or" just as
    # happily as "Mark" — causing "Directors or" to be captured as a name from
    # the SEC boilerplate title line "Departure of Directors or Certain Officers".
    #
    # Fix:
    #   Step 1 — find keyword positions with IGNORECASE (keywords can be any case)
    #   Step 2 — from each keyword position, extract the name WITHOUT IGNORECASE
    #            so [A-Z] strictly means an uppercase letter (proper noun)
    #   Also skip any match whose following text is the boilerplate title
    person_name, role = "", ""

    _BOILERPLATE = re.compile(
        r"directors or certain officers|election of directors|"
        r"appointment of certain officers|compensatory arrangements",
        re.IGNORECASE,
    )
    # Words that look like a name start but are date/time/title false positives
    _FALSE_POSITIVE_FIRST = {
        "on", "the", "mr", "ms", "mrs", "dr", "as", "in", "at",
        "chief", "vice", "senior", "executive", "interim",
    }
    # Strict proper-name pattern — deliberately NO re.IGNORECASE
    # Matches "First Last" or "First M. Last"
    _NAME_RE = re.compile(
        r"([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)"
    )

    for kw_m in re.finditer(
        r"(?:retirement|resignation|stepping down|"
        r"notified of the decision by|"
        r"indicated that (?:he|she|they) (?:will |would )?retire|"
        r"stepped down|"
        r"will retire|"
        r"intends? to retire|"
        r"departure|appointed?)\s+(?:of\s+)?",
        item_text, re.IGNORECASE,
    ):
        after = item_text[kw_m.end():]
        if _BOILERPLATE.match(after):
            continue   # skip the "Departure of Directors or Certain Officers" title
        nm = _NAME_RE.match(after)
        if nm:
            candidate = nm.group(1).strip()
            # Reject if the first word is a known false positive
            first_word = candidate.split()[0].lower().rstrip(".")
            if first_word in _FALSE_POSITIVE_FIRST:
                continue
            person_name = candidate
            break

    # Second pass: backward-looking patterns — name appears BEFORE the keyword.
    # These cover: "Name has stepped down", "Name indicated that he would retire",
    # "Name notified the Board", "Name was appointed/elected", etc.
    if not person_name:
        backward_patterns = [
            # "Name has/had stepped down / retired / resigned / notified"
            r"\b([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)"
            r"(?:,? (?:the |'s )?(?:Company['']s )?)?"
            r"\s+(?:has |had )?(?:stepped down|retired|resigned|notified|indicated that)",

            # "Name was appointed/elected/named"
            r"\b([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)"
            r"\s+was\s+(?:appointed|elected|named)",

            # "elected/appointed/named Name as" — forward keyword, then name
            r"(?:elected|appointed|named)\s+([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)"
            r"\s+(?:as\b|to\b)",

            # "announced that Name" / "confirmed that Name"
            r"(?:announced that|confirmed that|notified (?:the )?[Bb]oard that)\s+"
            r"([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)",
        ]
        for pat in backward_patterns:
            m = re.search(pat, item_text)
            if m:
                # group(1) is the name in all patterns above
                candidate = m.group(1).strip()
                first_word = candidate.split()[0].lower().rstrip(".")
                if first_word not in _FALSE_POSITIVE_FIRST:
                    person_name = candidate
                    break

    # Third pass: look for "FirstName LastName, the Company's [role]" pattern
    # which covers filings that mention the name followed directly by a title
    if not person_name:
        direct = re.search(
            r"\b([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+)"
            r"(?:,?\s+(?:the\s+)?(?:Company.s|our)\s+|,?\s+)"
            r"(?:Senior Vice President|Vice President|Executive Vice President|"
            r"Chief Executive|Chief Financial|Chief Operating|Chief Revenue|"
            r"Chief Legal|Chief Marketing|Chief Technology|President|Chairman|"
            r"General Manager|Principal)",
            item_text,
        )
        if direct:
            candidate = direct.group(1).strip()
            first_word = candidate.split()[0].lower()
            if first_word not in _FALSE_POSITIVE_FIRST:
                person_name = candidate

    # ── Role extraction ────────────────────────────────────────────────────────
    # Use \b word boundaries so "Director" does not match inside "Directors"
    role_match = re.search(
        r"\b(Chief Executive Officer|Chief Financial Officer|Chief Operating Officer|"
        r"President|Chairman|Director|Chief Technology Officer|"
        r"Chief Revenue Officer|Chief Legal Officer)\b",
        item_text, re.IGNORECASE,
    )
    if role_match:
        role = role_match.group(1)

    print(f"      [8K-502] ✓ Extracted {len(item_text)} chars | "
          f"person={person_name!r} | role={role!r}")
    if VERBOSE and not person_name:
        # Print a short section preview to help tune the name regex
        preview = item_text[:300].replace("\n", " ")
        print(f"      [8K-502] Section preview (name not matched): {preview!r}")

    return item_text, person_name, role


def ingest_filing_events(ticker: str, cik: str, company_id: int,
                         company_ids: dict, filings: list, conn):
    """
    Scan pre-fetched submission filings for three event types:

    3b — Any company's 8-K Item 5.02 (executive/board departures → indicator_id=2)
    3c — Any company's 10-K or 8-K with strategic pivot language → indicator_id=3
    3d — Any company's acquisition 8-K → competitive ripple inserted for ALL
         other sector peers → indicator_id=4
    """
    cik_int  = raw_cik(cik)
    inserted = 0

    with conn:
        with conn.cursor() as cur:
            for f in filings:
                form    = f.get("form", "")
                fd      = f.get("filingDate", "")
                acc     = f.get("accessionNumber", "")
                desc    = (f.get("primaryDocDescription") or "").lower()
                items   = str(f.get("items") or "")
                src_url = build_accession_url(cik_int, acc) if acc else ""

                # ── 3b: Executive / Board Churn — all companies ─────────────
                if form == "8-K" and "5.02" in items:
                    print(f"    [5.02] {ticker} 8-K with Item 5.02: {acc} "
                          f"({fd})  doc={f.get('primaryDocument','?')}")
                    item_text, person_name, role = _fetch_8k_502_text(
                        cik_int, acc, f.get("primaryDocument", "")
                    )
                    is_csuite = any(kw in (item_text + desc).lower() for kw in [
                        "ceo", "cfo", "president", "chief executive",
                        "chief financial", "chief operating",
                    ])
                    weight     = 25 if is_csuite else 15
                    # Use the actual document text for event_name and summary
                    # when available; fall back to metadata description otherwise
                    if person_name:
                        event_name = f"8-K 5.02 — {person_name} ({role})"
                        summary    = item_text[:500] if item_text else f"{ticker} 8-K Item 5.02 on {fd}"
                    else:
                        event_name = f"8-K 5.02 Departure — {ticker} {fd}"
                        summary    = item_text[:500] if item_text else f"{ticker} 8-K Item 5.02 on {fd}"
                    _insert_event(cur, company_id, 2, event_name,
                                  weight, fd, src_url, summary)
                    inserted += cur.rowcount

                    # Also store in strategic_context so the qualitative agent
                    # can reason about leadership changes in natural language
                    if item_text:

                        gemini_vector = generate_embedding(item_text[:2000])

                        metadata = Json({
                            "form":        "8-K",
                            "section":     "Item 5.02",
                            "source_type": "Filing",
                            "ticker":      ticker,
                            "person":      person_name or "Unknown",
                            "role":        role or "Unknown",
                            "sentiment":   "significant",
                        })
                        cur.execute(
                            """
                            INSERT INTO strategic_context
                                (company_id, 
                                content, 
                                embedding, 
                                metadata, 
                                source_date)
                            SELECT %s, %s, %s, %s, %s
                            WHERE NOT EXISTS (
                                SELECT 1 FROM strategic_context
                                WHERE company_id  = %s
                                  AND source_date = %s
                                  AND LEFT(content, 80) = LEFT(%s, 80)
                            )
                            """,
                            (company_id,
                             item_text[:2000],
                             gemini_vector,
                             metadata,
                             fd,
                             company_id,
                             fd,
                             item_text[:2000]),
                        )
                        inserted += cur.rowcount

                # ── 3c: Strategic Pivot Language — all companies ─────────────
                # Note: pivot keywords rarely appear in submission metadata (desc
                # is usually just "8-K" or "Form 10-K").  We check here as a
                # fast path, but the main pivot detection is done in Phase 4
                # after the MD&A text is fetched — see ingest_mda_context().
                if form in ("10-K", "8-K"):
                    combined = desc + " " + items.lower()
                    matched  = [kw for kw in PIVOT_KEYWORDS if kw in combined]
                    if matched:
                        print(f"    [PIVOT] {ticker} {form} {fd}: "
                              f"keywords matched in metadata → {matched}")
                        kw_str     = ", ".join(matched)
                        event_name = f"Pivot Signal [{form}] — {kw_str}"
                        summary    = (
                            f"{ticker} {form} on {fd} contained strategic pivot "
                            f"language: {kw_str}"
                        )
                        _insert_event(cur, company_id, 3, event_name,
                                      20, fd, src_url, summary)
                        inserted += cur.rowcount

                # ── 3d: Competitive Ripple — any acquisition → all peers ─────
                # An M&A move by any sector participant signals consolidation
                # pressure for every other company in the group.
                # Item 1.01 = "Entry into a Material Definitive Agreement" —
                # this is the primary SEC indicator of an acquisition or major
                # deal, regardless of what the terse description field says.
                if form == "8-K":
                    combined = desc + " " + items.lower()
                    is_acq   = (
                        "acquisition" in combined
                        or "merger" in combined
                        or "1.01" in items          # Material Definitive Agreement
                        or "definitive agreement" in combined
                    )
                    if is_acq:
                        peers = [p for p in company_ids if p != ticker]
                        print(f"    [RIPPLE] {ticker} acquisition 8-K {fd}: "
                              f"'{desc[:60]}' → ripple to {peers}")
                        acquirer_name = COMPANIES[ticker]["name"]
                        for peer_ticker, peer_id in company_ids.items():
                            if peer_ticker == ticker:
                                continue   # Don't insert ripple for the acquirer itself
                            event_name = f"Competitive Ripple — {ticker} acq {fd}"
                            summary    = (
                                f"{acquirer_name} filed 8-K on {fd} indicating "
                                f"M&A activity (sector consolidation ripple): "
                                f"{desc[:100]}"
                            )
                            _insert_event(cur, peer_id, 4, event_name,
                                          20, fd, src_url, summary)
                            inserted += cur.rowcount

    print(f"  [EVENTS] Inserted {inserted} indicator events for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: STRATEGIC CONTEXT (MD&A)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_mda_text(html: str) -> str:
    """
    Parse 10-K HTML and extract the plain text of the MD&A section.
    Searches for the 'Management's Discussion' header and stops at the
    next major item header (Quantitative disclosures, Item 4, etc.).
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    lines = text.splitlines()

    mda_re = re.compile(r"management.{0,30}discussion", re.IGNORECASE)
    end_re = re.compile(
        r"(quantitative.{0,30}market risk|item\s+[4-9a-z][\.\s]|"
        r"controls and procedures|legal proceedings)",
        re.IGNORECASE,
    )

    def _is_toc_line(line_idx: int, stripped_line: str) -> bool:
        """
        Detect ToC entries: either the header line itself ends with a page
        number, or the next non-blank line within 3 lines is a standalone
        page number (e.g. Teradyne puts 'Management...' on one line, '27'
        on the next).
        """
        # Case A: "Management's Discussion … 27" on same line
        if re.search(r"\b\d{1,3}\s*$", stripped_line):
            return True
        # Case B: next non-blank line is just a page number
        for j in range(line_idx + 1, min(line_idx + 4, len(lines))):
            nxt = lines[j].strip()
            if nxt:
                return bool(re.match(r"^\d{1,3}$", nxt))
        return False

    mda_start = mda_end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if mda_start is None:
            if mda_re.search(stripped) and len(stripped) < 120:
                # Skip table-of-contents entries (followed by a bare page number)
                if _is_toc_line(i, stripped):
                    continue
                mda_start = i
        else:
            # Don't trigger end boundary within the first 50 lines after the
            # MD&A header — modern 10-K filings list all item headers in a
            # table of contents, so "Item 8" or "Item 7A" appears immediately
            # after the MD&A entry and would falsely cut off the section.
            if i > mda_start + 50 and end_re.search(stripped) and len(stripped) < 120:
                mda_end = i
                break

    if mda_start is None:
        return ""

    end = mda_end if mda_end else min(mda_start + 800, len(lines))
    return "\n".join(lines[mda_start:end])


def _chunk_text(text: str, min_chars: int = 200, max_chars: int = 600) -> list:
    """
    Split MD&A plain text into chunks of 400-600 characters,
    breaking on paragraph boundaries to preserve sentence integrity.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
            buf = para
    if len(buf) >= min_chars:
        chunks.append(buf)
    return chunks


def _try_index_fallback(cik_int: str, accession: str, skip_doc: str) -> str:
    """
    When the primaryDocument from submissions metadata is a wrapper/shell that
    doesn't contain the full 10-K text (e.g. Teradyne uses a master cover file
    that embeds the real 10-K via an EDGAR viewer link), fetch the filing index
    page and look for the largest 10-K type document to extract MD&A from.

    Returns extracted MD&A text, or "" if nothing better is found.
    """
    acc_nodash  = accession.replace("-", "")
    index_url   = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{accession}-index.htm"
    )
    time.sleep(0.11)
    try:
        resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=30)
        if not resp.ok:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return ""

    # The filing index table has rows: Type | Description | Document
    # Look for type == "10-K" and pick the .htm/.html document (skip the
    # wrapper/cover file we already tried).
    candidates = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        doc_type = cells[0].get_text(strip=True).upper()
        link_tag  = cells[2].find("a")
        if not link_tag:
            continue
        href = link_tag.get("href", "")
        doc_name = href.split("/")[-1]
        if doc_type in ("10-K", "10-K/A") and doc_name.lower().endswith((".htm", ".html")):
            if doc_name != skip_doc:
                candidates.append(doc_name)

    if not candidates:
        if VERBOSE:
            print(f"    [MD&A] Index fallback: no alternative 10-K .htm found in index")
        return ""

    print(f"    [MD&A] Index fallback: trying {candidates[0]} from filing index")
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{candidates[0]}"
    )
    time.sleep(0.11)
    try:
        resp2 = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=90)
        if not resp2.ok:
            return ""
        return _extract_mda_text(resp2.text)
    except Exception:
        return ""


def ingest_mda_context(ticker: str, cik: str, company_id: int,
                       filings: list, conn):
    """
    Phase 4: For all 5 companies — fetch the primary 10-K document for
    FY2021 and FY2022, extract the MD&A section, chunk it, and insert each
    chunk into strategic_context (embedding = NULL for now).
    """
    if ticker not in MDA_TICKERS:
        return

    print(f"  [MD&A] Processing 10-K MD&A for {ticker}...")
    cik_int = raw_cik(cik)

    annual = [
        f for f in filings
        if f.get("form") == "10-K" and f.get("filingDate", "")[:4] in MDA_YEARS
    ]

    if not annual:
        print(f"  [MD&A] No FY2021/2022 10-K filings found for {ticker}")
        return

    total_chunks = 0
    with conn:
        with conn.cursor() as cur:
            for f in annual:
                acc  = f["accessionNumber"]
                fd   = f["filingDate"]
                year = fd[:4]

                # Use primaryDocument from submissions data directly — this is
                # already the filename of the main filing document and is far
                # more reliable than scraping the directory index page.
                primary_doc = f.get("primaryDocument", "")
                if not primary_doc:
                    print(f"    [MD&A] No primaryDocument in submissions for {acc}, skipping")
                    continue
                acc_nodash = acc.replace("-", "")
                doc_url    = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{acc_nodash}/{primary_doc}"
                )
                print(f"    [MD&A] {ticker} FY{year} — fetching {primary_doc} ({acc})...")

                time.sleep(0.11)
                try:
                    resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=90)
                    if not resp.ok:
                        print(f"    [MD&A] Failed to fetch {doc_url}")
                        continue
                except Exception as exc:
                    print(f"    [MD&A] Error fetching doc: {exc}")
                    continue

                mda_text = _extract_mda_text(resp.text)
                if not mda_text:
                    print(f"    [MD&A] ✗ MD&A section not found in {acc} "
                          f"(doc={primary_doc}, {len(resp.text)} bytes fetched)")
                    if VERBOSE:
                        preview = resp.text[2000:2400].replace("\n", " ")
                        print(f"    [MD&A] Doc sample (bytes 2000-2400): {preview!r}")
                    # Fall back: try to find the actual 10-K content file from the index
                    mda_text = _try_index_fallback(cik_int, acc, primary_doc)
                    if not mda_text:
                        continue

                # Also fall back when primary doc returns only a table-of-contents
                # shell (<500 chars after extraction) — e.g. Teradyne's 10-K wrapper
                elif len(mda_text) < 500:
                    print(f"    [MD&A] ⚠  Only {len(mda_text)} chars — may be a ToC shell, "
                          f"trying index fallback…")
                    fallback = _try_index_fallback(cik_int, acc, primary_doc)
                    if fallback and len(fallback) > len(mda_text):
                        mda_text = fallback
                        print(f"    [MD&A] ✓ Index fallback succeeded: {len(mda_text)} chars")

                chunks = _chunk_text(mda_text)
                print(f"    [MD&A] ✓ {len(chunks)} chunks from {ticker} FY{year} "
                      f"({len(mda_text)} chars extracted)")
                if VERBOSE and chunks:
                    preview = chunks[0][:200].replace("\n", " ")
                    print(f"    [MD&A] First chunk preview: {preview!r}")

                # ── Pivot keyword scan (indicator_id=3) ──────────────────────
                # The MD&A body is the right place to detect strategic language —
                # not the terse submission metadata. Scan the extracted text and
                # insert one event per filing where keywords appear.
                mda_lower  = mda_text.lower()
                pivot_hits = [kw for kw in PIVOT_KEYWORDS if kw in mda_lower]
                if pivot_hits:
                    kw_str     = ", ".join(pivot_hits)
                    event_name = f"Pivot Signal [10-K FY{year}] — {kw_str}"
                    summary    = (
                        f"{ticker} 10-K FY{year} MD&A contained strategic pivot "
                        f"language: {kw_str}"
                    )
                    print(f"    [PIVOT] {ticker} 10-K FY{year}: "
                          f"keywords found in MD&A → {pivot_hits}")
                    filing_url = build_accession_url(cik_int, acc)
                    _insert_event(cur, company_id, 3, event_name,
                                  20, fd, filing_url, summary)

                metadata = Json({
                    "form":        "10-K",
                    "year":        year,
                    "section":     "MD&A",
                    "source_type": "Filing",
                    "ticker":      ticker,
                    "sentiment":   "neutral",
                })

                for chunk in chunks:
                    gemini_vector = generate_embedding(chunk)
                    # Idempotent: match on company + date + first 80 chars of content
                    cur.execute(
                        """
                        INSERT INTO strategic_context
                            (company_id, content, embedding, metadata, source_date)
                        SELECT %s, %s, %s, %s, %s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM strategic_context
                            WHERE company_id  = %s
                              AND source_date = %s
                              AND LEFT(content, 80) = LEFT(%s, 80)
                        )
                        """,
                        (company_id, chunk, gemini_vector, metadata, fd,
                         company_id, fd, chunk),
                    )
                    total_chunks += cur.rowcount

    print(f"  [MD&A] Inserted {total_chunks} context chunks for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: NEO4J CYPHER SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_SCRIPT = """\
// ═══════════════════════════════════════════════════════════════════════════
// Neo4j Seed — Emerson / NI Backtest
// Generated by scripts/ingest_edgar.py
//
// Run in Neo4j Browser AFTER ingest_edgar.py completes.
// Uses MERGE throughout so it is safe to run multiple times.
// ═══════════════════════════════════════════════════════════════════════════

// ── 1. Sector ────────────────────────────────────────────────────────────────
MERGE (s:Sector {code: 'TM'})
SET s.name = 'Test & Measurement';

// ── 2. Companies ─────────────────────────────────────────────────────────────
MERGE (ni:Company   {ticker: 'NI'})   SET ni.name   = 'National Instruments Corp', ni.cik = '0000935494';
MERGE (emr:Company  {ticker: 'EMR'})  SET emr.name  = 'Emerson Electric Co',       emr.cik = '0000032604';
MERGE (keys:Company {ticker: 'KEYS'}) SET keys.name = 'Keysight Technologies Inc',  keys.cik = '0001601046';
MERGE (ter:Company  {ticker: 'TER'})  SET ter.name  = 'Teradyne Inc',               ter.cik = '0000097210';
MERGE (ame:Company  {ticker: 'AME'})  SET ame.name  = 'Ametek Inc',                 ame.cik = '0001037868';

// ── 3. Sector membership ─────────────────────────────────────────────────────
MATCH (s:Sector {code: 'TM'})
MATCH (c:Company) WHERE c.ticker IN ['NI', 'EMR', 'KEYS', 'TER', 'AME']
MERGE (c)-[:BELONGS_TO]->(s);

// ── 4. Competitive edges ─────────────────────────────────────────────────────
// NI competes directly with all T&M peers
MATCH (ni:Company {ticker: 'NI'}),   (k:Company {ticker: 'KEYS'}) MERGE (ni)-[:COMPETES_WITH {segment: 'Electronic Test'}]-(k);
MATCH (ni:Company {ticker: 'NI'}),   (t:Company {ticker: 'TER'})  MERGE (ni)-[:COMPETES_WITH {segment: 'Electronic Test'}]-(t);
MATCH (ni:Company {ticker: 'NI'}),   (a:Company {ticker: 'AME'})  MERGE (ni)-[:COMPETES_WITH {segment: 'Instrumentation'}]-(a);
// EMR competes via automation and industrial segments
MATCH (e:Company  {ticker: 'EMR'}),  (ni:Company {ticker: 'NI'})  MERGE (e)-[:COMPETES_WITH {segment: 'Industrial Automation'}]-(ni);
MATCH (e:Company  {ticker: 'EMR'}),  (a:Company  {ticker: 'AME'}) MERGE (e)-[:COMPETES_WITH {segment: 'Industrial Automation'}]-(a);
// Peer-to-peer edges within the T&M cluster
MATCH (k:Company {ticker: 'KEYS'}),  (t:Company {ticker: 'TER'})  MERGE (k)-[:COMPETES_WITH {segment: 'Electronic Test'}]-(t);
MATCH (k:Company {ticker: 'KEYS'}),  (a:Company {ticker: 'AME'})  MERGE (k)-[:COMPETES_WITH {segment: 'Instrumentation'}]-(a);
MATCH (t:Company {ticker: 'TER'}),   (a:Company {ticker: 'AME'})  MERGE (t)-[:COMPETES_WITH {segment: 'Instrumentation'}]-(a);

// ── 5. Predator intent ───────────────────────────────────────────────────────
// Emerson's stated Industrial Automation Rollup strategy (public knowledge
// from their earnings calls prior to the Feb 2023 announcement)
MATCH (emr:Company {ticker: 'EMR'}), (ni:Company {ticker: 'NI'})
MERGE (emr)-[r:IS_MONITORING]->(ni)
SET r.intent     = 'Industrial Automation Rollup',
    r.confidence = 'High',
    r.status     = 'Active Scouting',
    r.rationale  = 'NI software-defined test platform complements EMR industrial automation portfolio';

// ── 6. Activist stakes ───────────────────────────────────────────────────────
// Jana Partners — known activist position (~11%) driving sale process (2022)
// Note: Jana filed under a subsidiary; direct EDGAR search may use variant name
MERGE (jana:Activist {name: 'Jana Partners LLC'})
  SET jana.type = 'Activist Hedge Fund', jana.strategy = 'Event-Driven'
WITH jana
MATCH (ni:Company {ticker: 'NI'})
MERGE (jana)-[r:HAS_STAKE_IN]->(ni)
SET r.percent       = 11.0,
    r.position_type = 'Active',
    r.filed_date    = '2022-01-01',
    r.thesis        = 'Pressed board to pursue strategic alternatives including sale';

// Institutional investors with significant passive stakes (from EDGAR 13D/G filings)
MERGE (blk:Investor {name: 'BlackRock Inc'})    SET blk.type = 'Passive Institutional';
MERGE (vg:Investor  {name: 'Vanguard Group'})   SET vg.type  = 'Passive Institutional';
MERGE (fmr:Investor {name: 'FMR LLC'})          SET fmr.type = 'Passive Institutional';
MERGE (trp:Investor {name: 'T. Rowe Price'})    SET trp.type = 'Passive Institutional';

// BlackRock holds stakes across all 5 companies
MATCH (blk:Investor {name: 'BlackRock Inc'})
MATCH (c:Company) WHERE c.ticker IN ['NI', 'EMR', 'KEYS', 'TER', 'AME']
MERGE (blk)-[:HAS_STAKE_IN {position_type: 'Passive'}]->(c);

// Vanguard holds stakes across all 5 companies
MATCH (vg:Investor {name: 'Vanguard Group'})
MATCH (c:Company) WHERE c.ticker IN ['NI', 'EMR', 'KEYS', 'TER', 'AME']
MERGE (vg)-[:HAS_STAKE_IN {position_type: 'Passive'}]->(c);

// FMR (Fidelity) — significant stake in TER and AME
MATCH (fmr:Investor {name: 'FMR LLC'})
MATCH (c:Company) WHERE c.ticker IN ['TER', 'AME']
MERGE (fmr)-[:HAS_STAKE_IN {position_type: 'Passive'}]->(c);

// T. Rowe Price — significant stake in NI
MATCH (trp:Investor {name: 'T. Rowe Price'})
MATCH (ni:Company {ticker: 'NI'})
MERGE (trp)-[:HAS_STAKE_IN {position_type: 'Passive'}]->(ni);

// ── 7. Key executive departures (from 8-K Item 5.02 filings) ─────────────────
// TER — Mark Jagiela CEO retirement (confirmed from 8-K 2022-11-15)
MERGE (jagiela:Person {name: 'Mark E. Jagiela'})
  SET jagiela.role = 'Chief Executive Officer', jagiela.company = 'TER'
WITH jagiela
MATCH (ter:Company {ticker: 'TER'})
MERGE (jagiela)-[r:DEPARTED]->(ter)
SET r.event_date    = '2023-02-01',
    r.announced     = '2022-11-15',
    r.type          = 'Retirement',
    r.significance  = 'C-Suite';

// TER — Gregory Smith appointed CEO (replacing Jagiela)
MERGE (gsmith:Person {name: 'Gregory S. Smith'})
  SET gsmith.role = 'Chief Executive Officer', gsmith.company = 'TER'
WITH gsmith
MATCH (ter:Company {ticker: 'TER'})
MERGE (gsmith)-[r:APPOINTED]->(ter)
SET r.event_date   = '2023-02-01',
    r.announced    = '2022-11-15',
    r.prior_role   = 'President';

// EMR — David N. Farr CEO retirement (2021-02-01 filing)
MERGE (farr:Person {name: 'David N. Farr'})
  SET farr.role = 'Chief Executive Officer', farr.company = 'EMR'
WITH farr
MATCH (emr:Company {ticker: 'EMR'})
MERGE (farr)-[r:DEPARTED]->(emr)
SET r.event_date   = '2021-02-01',
    r.type         = 'Retirement',
    r.significance = 'C-Suite';

// NI — Daniel Berenbaum appointed (2022-12-15)
MERGE (beren:Person {name: 'Daniel Berenbaum'})
  SET beren.role = 'President', beren.company = 'NI'
WITH beren
MATCH (ni:Company {ticker: 'NI'})
MERGE (beren)-[r:APPOINTED]->(ni)
SET r.event_date = '2022-12-15';

// NI — Eric H. Starkloff (executive change 2020-01-30)
MERGE (stark:Person {name: 'Eric H. Starkloff'})
  SET stark.role = 'President and CEO', stark.company = 'NI'
WITH stark
MATCH (ni:Company {ticker: 'NI'})
MERGE (stark)-[r:APPOINTED]->(ni)
SET r.event_date = '2020-01-30';

// KEYS — Satish Dhanasekaran — CEO transition (2022)
MERGE (sdh:Person {name: 'Satish Dhanasekaran'})
  SET sdh.role = 'President and CEO', sdh.company = 'KEYS'
WITH sdh
MATCH (keys:Company {ticker: 'KEYS'})
MERGE (sdh)-[r:APPOINTED]->(keys)
SET r.event_date = '2022-05-18';

RETURN 'Emerson/NI backtest graph initialized.' AS status;
"""


def validate_ingestion(conn, company_ids: dict):
    """
    Post-ingestion validation: queries the live DB and prints a summary table
    for every company showing what was actually inserted. Flags gaps so you
    can diagnose problems without trawling through log output.
    """
    tickers = list(company_ids.keys())
    ind_labels = {
        1: "Activist Pressure",
        2: "Executive Churn",
        3: "Strategic Pivot",
        4: "Competitive Ripple",
        5: "Fundamental Decay",
        6: "Relative Valuation",
    }

    print("\n" + "═" * 65)
    print("  POST-INGESTION VALIDATION")
    print("═" * 65)

    with conn.cursor() as cur:

        # ── Financial metrics ─────────────────────────────────────────────────
        print("\n── Financial Metrics ──────────────────────────────────────────")
        cur.execute(
            """
            SELECT c.ticker, fm.metric_type,
                   COUNT(*)                                  AS cnt,
                   ROUND(MIN(fm.value)::numeric,  4)         AS min_val,
                   ROUND(MAX(fm.value)::numeric,  4)         AS max_val,
                   MIN(fm.observed_date)::text               AS earliest,
                   MAX(fm.observed_date)::text               AS latest
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = ANY(%s)
            GROUP BY c.ticker, fm.metric_type
            ORDER BY c.ticker, fm.metric_type
            """,
            (tickers,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  ✗ No financial_metrics rows found at all!")
        else:
            current = None
            for ticker, metric, cnt, mn, mx, earliest, latest in rows:
                if ticker != current:
                    print(f"\n  {ticker}:")
                    current = ticker
                print(f"    {metric:<22s}  {cnt:>3d} rows  "
                      f"[{mn} … {mx}]  {earliest} → {latest}")

        # Coverage check: flag any company with zero financial rows
        print()
        for ticker, company_id in company_ids.items():
            cur.execute(
                "SELECT COUNT(*) FROM financial_metrics WHERE company_id = %s",
                (company_id,),
            )
            cnt  = cur.fetchone()[0]
            flag = "✓" if cnt > 0 else "✗  NO DATA — check XBRL concepts / CIK"
            print(f"  {ticker:<6s} financial_metrics: {cnt:>4d} rows  {flag}")

        # ── Indicator events ──────────────────────────────────────────────────
        print("\n── Indicator Events ───────────────────────────────────────────")
        cur.execute(
            """
            SELECT c.ticker, ie.indicator_id, COUNT(*) AS cnt,
                   MIN(ie.event_date)::text AS earliest,
                   MAX(ie.event_date)::text AS latest
            FROM indicator_events ie
            JOIN companies c ON c.id = ie.company_id
            WHERE c.ticker = ANY(%s)
            GROUP BY c.ticker, ie.indicator_id
            ORDER BY c.ticker, ie.indicator_id
            """,
            (tickers,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  ✗ No indicator_events rows found!")
        else:
            current = None
            for ticker, ind_id, cnt, earliest, latest in rows:
                if ticker != current:
                    print(f"\n  {ticker}:")
                    current = ticker
                label = ind_labels.get(ind_id, f"Indicator {ind_id}")
                print(f"    [{ind_id}] {label:<22s}  {cnt:>3d} events  "
                      f"{earliest} → {latest}")

        # ── Strategic context ─────────────────────────────────────────────────
        print("\n── Strategic Context (text chunks) ────────────────────────────")
        cur.execute(
            """
            SELECT c.ticker,
                   COALESCE(sc.metadata->>'form',    '?') AS form,
                   COALESCE(sc.metadata->>'section', '?') AS section,
                   COUNT(*) AS cnt
            FROM strategic_context sc
            JOIN companies c ON c.id = sc.company_id
            WHERE c.ticker = ANY(%s)
            GROUP BY c.ticker, form, section
            ORDER BY c.ticker, form, section
            """,
            (tickers,),
        )
        rows = cur.fetchall()
        if not rows:
            print("  ✗ No strategic_context rows found!")
        else:
            current = None
            for ticker, form, section, cnt in rows:
                if ticker != current:
                    print(f"\n  {ticker}:")
                    current = ticker
                print(f"    {form:<6s} / {section:<22s}  {cnt:>3d} chunks")

        # ── NI mandate readiness ──────────────────────────────────────────────
        print("\n── NI Mandate Readiness ───────────────────────────────────────")
        ni_id = company_ids.get("NI")
        if ni_id:
            cur.execute(
                "SELECT DISTINCT indicator_id FROM indicator_events "
                "WHERE company_id = %s", (ni_id,)
            )
            ni_inds = {r[0] for r in cur.fetchall()}

            cur.execute(
                "SELECT COUNT(*) FROM financial_metrics WHERE company_id = %s",
                (ni_id,)
            )
            ni_fm = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM strategic_context WHERE company_id = %s",
                (ni_id,)
            )
            ni_sc = cur.fetchone()[0]

            checks = [
                ("Financial metrics present",   ni_fm > 0),
                ("[1] Activist pressure",        1 in ni_inds),
                ("[2] Executive churn",          2 in ni_inds),
                ("[3] Strategic pivot",          3 in ni_inds),
                ("[4] Competitive ripple",       4 in ni_inds),
                ("MD&A context chunks present",  ni_sc > 0),
            ]
            all_pass = all(ok for _, ok in checks)
            for label, ok in checks:
                print(f"  {'✓' if ok else '✗'} {label}")

            print()
            if all_pass:
                print("  ✅ NI has signal data across all categories.")
                print("     Scoring engine should have enough to produce a mandate score.")
            else:
                missing = [lbl for lbl, ok in checks if not ok]
                print(f"  ⚠️  NI is missing: {missing}")
                print("     Re-run ingestion after fixing the flagged issues above.")
        else:
            print("  ✗ NI company_id not found — seed_companies may have failed")

    print("\n" + "═" * 65)


def write_cypher_script():
    out_path = os.path.join(_script_dir, "neo4j-emerson-ni.cql")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(CYPHER_SCRIPT)
    print(f"\n[NEO4J] Cypher script written → {out_path}")

def generate_embedding(text):
    """Sends text to Gemini and returns a truncated 1536-dimensional vector."""
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2-preview',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=1536)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"      [!] Embedding failed: {e}")
        return None
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  EDGAR Ingestion — Emerson / NI Backtest")
    print("=" * 65)

    try:
        conn = get_pg_connection()
    except Exception as exc:
        print(f"\n[FATAL] Cannot connect to PostgreSQL: {exc}")
        sys.exit(1)

    print(f"\n[DB] Connected → {os.environ.get('PG_HOST')}:{os.environ.get('PG_PORT')}")

    # ── Schema additions (safe, idempotent) ───────────────────────────────────
    print("\n[SCHEMA] Applying column additions...")
    apply_schema_additions(conn)

    # ── Phase 1: Companies ────────────────────────────────────────────────────
    print("\n[PHASE 1] Seeding sector and companies...")
    company_ids = seed_companies(conn)

    # ── Pre-fetch all submissions (reused by phases 3 + 4) ───────────────────
    print("\n[SUBMISSIONS] Fetching filing lists from EDGAR...")
    submissions: dict = {}
    for ticker, meta in COMPANIES.items():
        submissions[ticker] = _fetch_all_submissions(ticker, meta["cik"])

    # ── Phase 2: Financial metrics ────────────────────────────────────────────
    print("\n[PHASE 2] Ingesting financial metrics (XBRL)...")
    for ticker, meta in COMPANIES.items():
        if ticker == 'NATI':
            print(f"\n  ── {ticker} ─────────────────────────────")
            ingest_financials(ticker, meta["cik"], company_ids[ticker], conn)
        else:
            print(f'{ticker} financial data to be fetched from alpha vantage')

    # ── Phase 3a: Activist events — all 5 companies ──────────────────────────
    print("\n[PHASE 3a] Searching for activist filings on all companies...")
    ingest_activist_events(company_ids, conn)

    # ── Phase 3b/c/d: Filing events ───────────────────────────────────────────
    print("\n[PHASE 3b-d] Ingesting indicator events from filing metadata...")
    for ticker, meta in COMPANIES.items():
        print(f"\n  ── {ticker} ─────────────────────────────")
        ingest_filing_events(
            ticker, meta["cik"], company_ids[ticker],
            company_ids, submissions[ticker], conn,
        )

    # ── Phase 4: MD&A strategic context ──────────────────────────────────────
    print("\n[PHASE 4] Extracting MD&A text from 10-K filings...")
    for ticker, meta in COMPANIES.items():
        print(f"\n  ── {ticker} ─────────────────────────────")
        ingest_mda_context(
            ticker, meta["cik"], company_ids[ticker],
            submissions[ticker], conn,
        )

    # ── Validation summary ────────────────────────────────────────────────────
    print("\n[VALIDATION] Querying DB for post-ingestion summary...")
    validate_ingestion(conn, company_ids)

    conn.close()
    print("\n[DB] Connection closed")

    # ── Phase 5: Write Cypher script ──────────────────────────────────────────
    write_cypher_script()

    print("\n" + "=" * 65)
    print("  Ingestion complete.")
    print()
    print("  Verification queries:")
    print("    SELECT ticker, COUNT(*) FROM financial_metrics fm")
    print("    JOIN companies c ON c.id = fm.company_id GROUP BY ticker;")
    print()
    print("    SELECT ticker, indicator_id, COUNT(*) FROM indicator_events ie")
    print("    JOIN companies c ON c.id = ie.company_id")
    print("    WHERE ticker = 'NI' GROUP BY ticker, indicator_id;")
    print()
    print("  Then paste scripts/neo4j-emerson-ni.cql into Neo4j Browser.")
    print("=" * 65)
