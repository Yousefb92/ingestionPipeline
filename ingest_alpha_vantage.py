#!/usr/bin/env python3
"""
Alpha Vantage Financial Ingestion Script
════════════════════════════════════════
Replaces the XBRL-based financial metrics ingestion with clean, pre-calculated
figures from the Alpha Vantage Premium API.

Why Alpha Vantage instead of SEC EDGAR XBRL?
  - EDGAR XBRL for large conglomerates (EMR) uses fallback concepts like
    NetIncomeLoss instead of OperatingIncomeLoss, producing wildly wrong margins.
  - EDGAR quarterly EBITDA is cumulative YTD, not standalone — NATI Q1-2020
    showed 61.77% because the YTD figure was used as a single-quarter margin.
  - Alpha Vantage returns pre-calculated, standalone `ebitda` values that are
    auditor-reconciled and period-correct.

Metrics inserted per company per period:
  ebitda_margin       — EBITDA / Revenue
  gross_margin        — Gross Profit / Revenue
  rd_intensity        — R&D Expense / Revenue  (key signal for T&M acquirers)
  fcf_margin          — (Operating CF - CapEx) / Revenue
  net_debt_mm         — Total Debt - Cash, in $M  (leverage signal)
  revenue_mm          — Absolute revenue in $M  (size context)
  enterprise_value_mm — Most recent EV from OVERVIEW endpoint

Usage:
    cd C:/Users/Yousef/PycharmProjects/ingestion
    python scripts/ingest_alpha_vantage.py

Prerequisites:
    Add ALPHA_VANTAGE_API_KEY to your .env file:
        ALPHA_VANTAGE_API_KEY=<your_premium_key>

The script is idempotent — safe to run multiple times. Existing rows are
skipped (WHERE NOT EXISTS on company_id + metric_type + observed_date).
"""

import os
import sys
import time
import math
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date
from dotenv import load_dotenv

import requests

# ── .env resolution ───────────────────────────────────────────────────────────
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)               # ingestion/
_sibling_env = os.path.join(_project_dir, "..", "ma_agent_task", ".env")

load_dotenv(dotenv_path=os.path.join(_project_dir, ".env"))   # ingestion/.env
load_dotenv(dotenv_path=_sibling_env)                          # ma_agent_task/.env (fallback)

AV_API_KEY = ""
if not AV_API_KEY:
    print("❌  ALPHA_VANTAGE_API_KEY not set in .env. Add it and re-run.")
    sys.exit(1)

AV_BASE = "https://www.alphavantage.co/query"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COMPANIES = {
    "NATI":  "National Instruments Corp",
    "EMR":   "Emerson Electric Co",
    "KEYS":  "Keysight Technologies Inc",
    "TER":   "Teradyne Inc",
    "AME":   "Ametek Inc",
}

# Only ingest data within this window (matches the scoring engine backtest date)
DATE_FROM = date(2019, 1, 1)
DATE_TO   = date(2022, 4, 12)

# Sleep between API calls. Premium key supports 75 req/min → 0.8 s is safe.
# Drop to 0.2 s if you have a higher-tier key. Free tier: set to 13 s.
AV_SLEEP = 0.8

# ── EDGAR fallback for delisted companies ─────────────────────────────────────
# When Alpha Vantage returns no income data (company delisted), fall back to
# SEC EDGAR XBRL companyfacts. Add the CIK here for any delisted ticker.
EDGAR_FALLBACK_CIKS = {
    "NATI": "0000935494",   # National Instruments — acquired by Emerson Jan 2023
}

# Known historical market caps (in $M) at DATE_TO for delisted companies.
# Used for enterprise_value_mm when the OVERVIEW endpoint returns nothing.
# Source: Bloomberg/Yahoo Finance historical close × shares outstanding.
MANUAL_EV_MM = {
    "NATI": 6_800.0,   # ~$6.8B market cap as of April 2022 (pre-announcement)
}

EDGAR_BASE    = "https://data.sec.gov"
EDGAR_HEADERS = {"User-Agent": "PwC-CaseStudy ingestion@pwc.com"}


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


def _insert_metric(cur, company_id: int, metric_type: str, value: float,
                   end_date: str, doc_label: str):
    """Idempotent upsert for a single financial_metrics row."""
    cur.execute(
        """
        INSERT INTO financial_metrics
            (company_id, metric_type, value, observed_date, source_document)
        SELECT %s, %s, %s, %s, %s
        WHERE NOT EXISTS (
            SELECT 1 FROM financial_metrics
            WHERE company_id    = %s
              AND metric_type   = %s
              AND observed_date = %s
        )
        """,
        (company_id, metric_type, value, end_date, doc_label,
         company_id, metric_type, end_date),
    )
    return cur.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA VANTAGE HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def av_get(function: str, symbol: str, extra_params: dict = None) -> dict | None:
    """
    Calls the Alpha Vantage API and returns the parsed JSON dict.
    Returns None on any error. Respects AV_SLEEP rate limiting.
    """
    params = {
        "function": function,
        "symbol":   symbol,
        "apikey":   AV_API_KEY,
    }
    if extra_params:
        params.update(extra_params)

    time.sleep(AV_SLEEP)
    try:
        resp = requests.get(AV_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # AV returns {"Note": "..."} or {"Information": "..."} on rate limit / bad key
        if "Note" in data:
            print(f"  ⚠  [AV] Rate limit hit for {function}/{symbol}: {data['Note'][:80]}")
            time.sleep(60)  # Back off for a full minute
            return None
        if "Information" in data:
            print(f"  ⚠  [AV] API message for {function}/{symbol}: {data['Information'][:120]}")
            return None

        return data

    except Exception as exc:
        print(f"  ❌  [AV] Request failed {function}/{symbol}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# VALUE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse(value, default=None):
    """
    Alpha Vantage returns numeric strings or the literal string "None".
    Converts to float, or returns `default` if the value is absent/None/invalid.
    """
    if value is None or value == "None":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _in_range(fiscal_date_str: str) -> bool:
    """Returns True if the fiscal period end falls within DATE_FROM..DATE_TO."""
    try:
        d = date.fromisoformat(fiscal_date_str)
        return DATE_FROM <= d <= DATE_TO
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: INCOME STATEMENT METRICS
# ─────────────────────────────────────────────────────────────────────────────

def ingest_income_statement(ticker: str, company_id: int, conn) -> dict:
    """
    Fetches INCOME_STATEMENT from Alpha Vantage and inserts:
      ebitda_margin, gross_margin, rd_intensity, revenue_mm

    Returns a dict of {fiscal_date: {revenue, ebitda}} for use in FCF margin
    calculation (which divides by revenue from the income statement).
    """
    print(f"  [INCOME] Fetching income statement for {ticker}...")
    data = av_get("INCOME_STATEMENT", ticker)
    if not data:
        return {}

    revenue_by_date = {}  # {fiscal_date_str: revenue_float} for FCF calc
    inserted = 0

    with conn:
        with conn.cursor() as cur:
            for report in data.get("annualReports", []):
                fiscal_date = report.get("fiscalDateEnding", "")
                if not _in_range(fiscal_date):
                    continue

                revenue   = _parse(report.get("totalRevenue"))
                ebitda    = _parse(report.get("ebitda"))
                gross_p   = _parse(report.get("grossProfit"))
                rd        = _parse(report.get("researchAndDevelopment"))

                if not revenue or revenue == 0:
                    print(f"    ⚠  {ticker} {fiscal_date}: no revenue, skipping period")
                    continue

                label = f"AV Annual FY{fiscal_date[:4]}"
                revenue_by_date[fiscal_date] = revenue

                # 1. EBITDA margin
                if ebitda is not None:
                    margin = ebitda / revenue
                    # Sanity gate: real industrials sit in roughly -0.1 to 0.5
                    if not (-0.1 <= margin <= 0.5):
                        print(f"    ⚠  [SANITY] {ticker} {fiscal_date}: "
                              f"ebitda_margin={margin:.3f} — check AV data "
                              f"(ebitda={ebitda/1e6:.1f}M, rev={revenue/1e6:.1f}M)")
                    inserted += _insert_metric(cur, company_id, "ebitda_margin",
                                               margin, fiscal_date, label)
                    print(f"    ✓  {ticker} {fiscal_date}: ebitda_margin={margin:.4f} "
                          f"({ebitda/1e6:.1f}M / {revenue/1e6:.1f}M)")
                else:
                    print(f"    ⚠  {ticker} {fiscal_date}: ebitda not available from AV")

                # 2. Gross margin
                if gross_p is not None:
                    inserted += _insert_metric(cur, company_id, "gross_margin",
                                               gross_p / revenue, fiscal_date, label)

                # 3. R&D intensity
                if rd is not None:
                    inserted += _insert_metric(cur, company_id, "rd_intensity",
                                               rd / revenue, fiscal_date, label)

                # 4. Revenue in $M
                inserted += _insert_metric(cur, company_id, "revenue_mm",
                                           revenue / 1_000_000, fiscal_date, label)

    print(f"  [INCOME] Inserted {inserted} income metric rows for {ticker}")
    return revenue_by_date


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: BALANCE SHEET METRICS
# ─────────────────────────────────────────────────────────────────────────────

def ingest_balance_sheet(ticker: str, company_id: int, conn):
    """
    Fetches BALANCE_SHEET and inserts net_debt_mm.
    Net debt = Total Debt - Cash & Short-Term Investments.
    """
    print(f"  [BALANCE] Fetching balance sheet for {ticker}...")
    data = av_get("BALANCE_SHEET", ticker)
    if not data:
        return

    inserted = 0
    with conn:
        with conn.cursor() as cur:
            for report in data.get("annualReports", []):
                fiscal_date = report.get("fiscalDateEnding", "")
                if not _in_range(fiscal_date):
                    continue

                # Use total debt (short + long term) when available, else long-term only
                total_debt = _parse(report.get("shortLongTermDebtTotal"))
                long_debt  = _parse(report.get("longTermDebt"))
                cash       = _parse(report.get("cashAndShortTermInvestments")) or \
                             _parse(report.get("cashAndCashEquivalentsAtCarryingValue")) or 0.0

                debt = total_debt if total_debt is not None else long_debt
                if debt is None:
                    print(f"    ⚠  {ticker} {fiscal_date}: no debt data available")
                    continue

                net_debt_mm = (debt - cash) / 1_000_000
                label = f"AV Balance Sheet FY{fiscal_date[:4]}"
                inserted += _insert_metric(cur, company_id, "net_debt_mm",
                                           net_debt_mm, fiscal_date, label)
                print(f"    ✓  {ticker} {fiscal_date}: net_debt_mm={net_debt_mm:.1f}M "
                      f"(debt={debt/1e6:.1f}M, cash={cash/1e6:.1f}M)")

    print(f"  [BALANCE] Inserted {inserted} balance sheet rows for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: CASH FLOW METRICS
# ─────────────────────────────────────────────────────────────────────────────

def ingest_cash_flow(ticker: str, company_id: int, conn, revenue_by_date: dict):
    """
    Fetches CASH_FLOW and inserts fcf_margin.
    FCF = Operating Cash Flow - Capital Expenditures.
    Divides by revenue (passed in from the income statement phase).
    """
    print(f"  [CASHFLOW] Fetching cash flow for {ticker}...")
    data = av_get("CASH_FLOW", ticker)
    if not data:
        return

    inserted = 0
    with conn:
        with conn.cursor() as cur:
            for report in data.get("annualReports", []):
                fiscal_date = report.get("fiscalDateEnding", "")
                if not _in_range(fiscal_date):
                    continue
                if fiscal_date not in revenue_by_date:
                    continue  # No matching revenue to divide by

                cfo   = _parse(report.get("operatingCashflow"))
                capex = _parse(report.get("capitalExpenditures"))

                if cfo is None or capex is None:
                    continue

                revenue = revenue_by_date[fiscal_date]
                if not revenue:
                    continue

                # CapEx is reported as a negative number by AV in most cases.
                # abs() ensures the subtraction always works correctly.
                fcf = cfo - abs(capex)
                fcf_margin = fcf / revenue

                label = f"AV Cash Flow FY{fiscal_date[:4]}"
                inserted += _insert_metric(cur, company_id, "fcf_margin",
                                           fcf_margin, fiscal_date, label)
                print(f"    ✓  {ticker} {fiscal_date}: fcf_margin={fcf_margin:.4f} "
                      f"(cfo={cfo/1e6:.1f}M, capex={abs(capex)/1e6:.1f}M)")

    print(f"  [CASHFLOW] Inserted {inserted} cash flow rows for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: ENTERPRISE VALUE (historical, as of DATE_TO)
# ─────────────────────────────────────────────────────────────────────────────

def _get_historical_price(ticker: str, target_date: date) -> float | None:
    """
    Returns the closing price on target_date (or the nearest prior trading day)
    using AV's TIME_SERIES_DAILY_ADJUSTED endpoint.

    Uses outputsize=full to ensure coverage of dates years in the past.
    Falls back up to 5 trading days earlier if the exact date is not present
    (e.g. the date falls on a weekend or market holiday).
    """
    print(f"  [EV] Fetching historical daily prices for {ticker}...")
    data = av_get("TIME_SERIES_DAILY_ADJUSTED", ticker,
                  extra_params={"outputsize": "full"})
    if not data:
        return None

    series = data.get("Time Series (Daily)", {})
    if not series:
        print(f"    ⚠  {ticker}: no daily price series returned")
        return None

    # Walk back up to 5 trading days to find the nearest available price
    from datetime import timedelta
    for offset in range(6):
        lookup = (target_date - timedelta(days=offset)).isoformat()
        if lookup in series:
            price = float(series[lookup]["4. close"])
            print(f"    ✓  {ticker}: closing price on {lookup} = ${price:.2f}"
                  + (f" (offset -{offset}d from {target_date})" if offset else ""))
            return price

    print(f"    ⚠  {ticker}: no price found within 5 days of {target_date}")
    return None


def ingest_enterprise_value(ticker: str, company_id: int, conn):
    """
    Calculates enterprise_value_mm as of DATE_TO using:
        Market Cap = historical closing price × shares outstanding

    This avoids the future-data bias of AV's OVERVIEW endpoint, which returns
    the current live market cap rather than the historical value at the backtest date.

    Shares outstanding come from OVERVIEW (they change slowly and rarely cause
    material errors over a 1–2 year window). Price is sourced from the daily
    adjusted price series at DATE_TO.
    """
    # Step 1: Get shares outstanding from OVERVIEW (structural figure, changes slowly)
    print(f"  [EV] Fetching share count from OVERVIEW for {ticker}...")
    overview = av_get("OVERVIEW", ticker)
    if not overview:
        print(f"    ⚠  {ticker}: OVERVIEW unavailable — skipping EV")
        return

    shares = _parse(overview.get("SharesOutstanding"))
    if shares is None:
        print(f"    ⚠  {ticker}: no SharesOutstanding in OVERVIEW")
        return

    # Step 2: Get the historical closing price on DATE_TO
    price = _get_historical_price(ticker, DATE_TO)
    if price is None:
        return

    # Step 3: Market cap as of DATE_TO = price × shares
    market_cap_mm = (price * shares) / 1_000_000
    label = f"AV historical market cap at {DATE_TO} (price × shares)"

    inserted = 0
    with conn:
        with conn.cursor() as cur:
            inserted += _insert_metric(cur, company_id, "enterprise_value_mm",
                                       market_cap_mm, DATE_TO.isoformat(), label)
    print(f"  [EV] {ticker}: enterprise_value_mm={market_cap_mm:,.1f}M "
          f"(${price:.2f} × {shares/1e6:.1f}M shares)")
    print(f"  [EV] Inserted {inserted} EV rows for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR XBRL FALLBACK  (used automatically for delisted tickers)
# ─────────────────────────────────────────────────────────────────────────────

def _edgar_get(url: str):
    """Rate-limited GET to EDGAR (10 req/s limit → 0.11 s sleep)."""
    time.sleep(0.11)
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        if "json" in resp.headers.get("Content-Type", ""):
            return resp.json()
        return resp.text
    except Exception as exc:
        print(f"    [EDGAR] Request failed {url}: {exc}")
        return None


def _edgar_concept_series(facts_usgaap: dict, concepts: list) -> dict:
    """
    Tries each XBRL concept in order; returns the first that has ≥ 1 annual
    (10-K) entry in the DATE_FROM..DATE_TO window.
    Returns {end_date_str: value_float}.
    """
    for concept in concepts:
        node = facts_usgaap.get(concept)
        if not node:
            continue
        entries = node.get("units", {}).get("USD", [])
        series = {}
        for e in entries:
            if e.get("form") != "10-K":       # Annual filings only — avoids YTD quarterly issue
                continue
            end = e.get("end", "")
            try:
                end_date = date.fromisoformat(end)
            except ValueError:
                continue
            if DATE_FROM <= end_date <= DATE_TO:
                series[end] = float(e.get("val", 0) or 0)
        if series:
            print(f"      [EDGAR XBRL] '{concept}' ✓  {len(series)} annual periods in range")
            return series
        else:
            print(f"      [EDGAR XBRL] '{concept}' — present but no 10-K entries in {DATE_FROM}…{DATE_TO}")
    return {}


def ingest_financials_from_edgar(ticker: str, cik: str, company_id: int, conn) -> dict:
    """
    EDGAR XBRL fallback for delisted companies.
    Inserts the same metric types as the Alpha Vantage phases:
      ebitda_margin, gross_margin, rd_intensity, fcf_margin, net_debt_mm, revenue_mm

    Uses annual 10-K filings only to avoid cumulative-YTD quarterly issues.
    Returns {fiscal_date: revenue_float} for downstream use (matches AV signature).
    """
    print(f"  [EDGAR FALLBACK] Fetching XBRL companyfacts for {ticker} (CIK {cik})...")
    cik_padded = cik.zfill(10) if not cik.startswith("0") else cik
    data = _edgar_get(f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json")
    if not data or not isinstance(data, dict):
        print(f"  [EDGAR FALLBACK] No XBRL data returned for {ticker}.")
        return {}

    facts = data.get("facts", {}).get("us-gaap", {})
    if not facts:
        print(f"  [EDGAR FALLBACK] No us-gaap facts for {ticker}.")
        return {}

    # Concept resolution — ordered by preference
    revenues    = _edgar_concept_series(facts, [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "NetRevenues",
    ])
    opincome    = _edgar_concept_series(facts, [
        "OperatingIncomeLoss",           # Should be directly available for NATI
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ])
    da_vals     = _edgar_concept_series(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ])
    grossprofit = _edgar_concept_series(facts, ["GrossProfit"])
    rd_exp      = _edgar_concept_series(facts, ["ResearchAndDevelopmentExpense"])
    cfo         = _edgar_concept_series(facts, ["NetCashProvidedByUsedInOperatingActivities"])
    capex       = _edgar_concept_series(facts, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ])
    debt        = _edgar_concept_series(facts, [
        "LongTermDebtNoncurrent", "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
    ])

    ebitda_dates = set(revenues) & set(da_vals) & set(opincome)
    print(f"  [EDGAR FALLBACK] {len(ebitda_dates)} aligned EBITDA periods for {ticker}")

    revenue_by_date = {}
    inserted = 0
    cik_int = str(int(cik))

    with conn:
        with conn.cursor() as cur:
            for end_date in sorted(ebitda_dates):
                rev = revenues[end_date]
                da  = da_vals[end_date]
                oi  = opincome[end_date]
                if not rev:
                    continue

                label = f"EDGAR 10-K FY{end_date[:4]}"
                revenue_by_date[end_date] = rev

                # 1. EBITDA margin
                ebitda_margin = (oi + da) / rev
                if not (-0.1 <= ebitda_margin <= 0.5):
                    print(f"    ⚠  [SANITY] {ticker} {end_date}: ebitda_margin={ebitda_margin:.3f} "
                          f"(oi={oi/1e6:.1f}M, da={da/1e6:.1f}M, rev={rev/1e6:.1f}M)")
                inserted += _insert_metric(cur, company_id, "ebitda_margin",
                                           ebitda_margin, end_date, label)
                print(f"    ✓  {ticker} {end_date}: ebitda_margin={ebitda_margin:.4f} "
                      f"({(oi+da)/1e6:.1f}M EBITDA / {rev/1e6:.1f}M rev)")

                # 2. Gross margin
                if end_date in grossprofit:
                    inserted += _insert_metric(cur, company_id, "gross_margin",
                                               grossprofit[end_date] / rev, end_date, label)

                # 3. R&D intensity
                if end_date in rd_exp:
                    inserted += _insert_metric(cur, company_id, "rd_intensity",
                                               rd_exp[end_date] / rev, end_date, label)

                # 4. FCF margin
                if end_date in cfo and end_date in capex:
                    fcf = cfo[end_date] - abs(capex[end_date])
                    inserted += _insert_metric(cur, company_id, "fcf_margin",
                                               fcf / rev, end_date, label)

                # 5. Revenue in $M
                inserted += _insert_metric(cur, company_id, "revenue_mm",
                                           rev / 1_000_000, end_date, label)

            # 6. Net debt — uses its own date set (balance sheet dates may differ)
            for end_date, debt_val in sorted(debt.items()):
                label = f"EDGAR 10-K BS FY{end_date[:4]}"
                inserted += _insert_metric(cur, company_id, "net_debt_mm",
                                           debt_val / 1_000_000, end_date, label)

    print(f"  [EDGAR FALLBACK] Inserted {inserted} metric rows for {ticker}")
    return revenue_by_date


def ingest_ev_manual(ticker: str, company_id: int, conn):
    """
    Inserts a manually specified enterprise_value_mm for delisted companies
    that return nothing from the OVERVIEW endpoint.
    Values are sourced from historical market data at DATE_TO.
    """
    ev_mm = MANUAL_EV_MM.get(ticker)
    if ev_mm is None:
        print(f"  [EV MANUAL] No manual EV configured for {ticker} — skipping")
        return

    inserted = 0
    with conn:
        with conn.cursor() as cur:
            inserted += _insert_metric(
                cur, company_id, "enterprise_value_mm",
                ev_mm, DATE_TO.isoformat(),
                f"Manual historical market cap at {DATE_TO}"
            )
    print(f"  [EV MANUAL] {ticker}: enterprise_value_mm={ev_mm:,.1f}M inserted ({inserted} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: CLEAR OLD EDGAR-DERIVED METRICS (optional, run once)
# ─────────────────────────────────────────────────────────────────────────────

def clear_edgar_financials(ticker: str, company_id: int, conn):
    """
    Removes all financial_metrics rows that were ingested from EDGAR XBRL
    (identified by source_document starting with '10-K' or '10-Q').
    Call this BEFORE running the Alpha Vantage ingestion to avoid stale data.

    Set DRY_RUN = True to preview what would be deleted without committing.
    """
    DRY_RUN = False  # Flip to True to preview

    count_query = """
        SELECT COUNT(*) FROM financial_metrics
        WHERE company_id = %s
          AND (source_document LIKE '10-K%%' OR source_document LIKE '10-Q%%')
    """
    delete_query = """
        DELETE FROM financial_metrics
        WHERE company_id = %s
          AND (source_document LIKE '10-K%%' OR source_document LIKE '10-Q%%')
    """

    with conn.cursor() as cur:
        cur.execute(count_query, (company_id,))
        count = cur.fetchone()[0]
        print(f"  [CLEAR] {ticker}: {count} EDGAR-sourced metric rows found")
        if count > 0 and not DRY_RUN:
            cur.execute(delete_query, (company_id,))
            conn.commit()
            print(f"  [CLEAR] {ticker}: {count} rows deleted ✓")
        elif DRY_RUN:
            print(f"  [CLEAR] DRY_RUN=True — no rows deleted")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(ev_only: bool = False):
    mode = "EV-only" if ev_only else "full"
    print("\n" + "="*60)
    print(f"  Alpha Vantage Financial Ingestion  [{mode}]")
    print(f"  Window: {DATE_FROM} → {DATE_TO}")
    print("="*60 + "\n")

    conn = get_pg_connection()

    # Fetch company IDs from the database
    company_ids = {}
    with conn.cursor() as cur:
        for ticker in COMPANIES:
            cur.execute("SELECT id FROM companies WHERE ticker = %s", (ticker,))
            row = cur.fetchone()
            if row:
                company_ids[ticker] = row[0]
                print(f"  [SETUP] {ticker} → company_id={row[0]}")
            else:
                print(f"  ⚠  [SETUP] {ticker} not found in companies table — run ingest_edgar.py first")

    if not company_ids:
        print("❌  No companies found. Run ingest_edgar.py first to seed company rows.")
        conn.close()
        sys.exit(1)

    for ticker, company_id in company_ids.items():
        print(f"\n{'─'*60}")
        print(f"  Processing {ticker} (company_id={company_id})")
        print(f"{'─'*60}")

        if ev_only:
            # ── EV-only mode: delete existing EV row and re-insert from history ──
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM financial_metrics "
                        "WHERE company_id = %s AND metric_type = 'enterprise_value_mm'",
                        (company_id,)
                    )
                    print(f"  [CLEAR] Removed existing enterprise_value_mm for {ticker}")
            if ticker in EDGAR_FALLBACK_CIKS:
                ingest_ev_manual(ticker, company_id, conn)
            else:
                ingest_enterprise_value(ticker, company_id, conn)
            print(f"  ✅  {ticker} EV updated")
            continue

        # ── Full ingestion mode ───────────────────────────────────────────────

        # Step 1: Clear old stale metrics for this company
        clear_edgar_financials(ticker, company_id, conn)

        # Step 2: Try Alpha Vantage first — auto-detect delisted companies by
        #         empty return, then fall back to EDGAR XBRL
        revenue_by_date = ingest_income_statement(ticker, company_id, conn)

        if not revenue_by_date and ticker in EDGAR_FALLBACK_CIKS:
            print(f"  ⚠  [FALLBACK] AV returned no income data for {ticker} "
                  f"(likely delisted). Switching to EDGAR XBRL...")
            revenue_by_date = ingest_financials_from_edgar(
                ticker, EDGAR_FALLBACK_CIKS[ticker], company_id, conn
            )
        elif not revenue_by_date:
            print(f"  ⚠  {ticker}: no income data from AV and no EDGAR fallback configured")

        # Step 3: Balance sheet (net_debt_mm) — AV only; EDGAR fallback handles
        #         net_debt inside ingest_financials_from_edgar above
        if ticker not in EDGAR_FALLBACK_CIKS:
            ingest_balance_sheet(ticker, company_id, conn)

        # Step 4: Cash flow (fcf_margin) — AV only (EDGAR fallback includes FCF)
        if ticker not in EDGAR_FALLBACK_CIKS:
            ingest_cash_flow(ticker, company_id, conn, revenue_by_date)

        # Step 5: Enterprise value — use manual historical value for delisted companies
        if ticker in EDGAR_FALLBACK_CIKS:
            ingest_ev_manual(ticker, company_id, conn)
        else:
            ingest_enterprise_value(ticker, company_id, conn)

        print(f"  ✅  {ticker} complete")

    conn.close()
    print(f"\n{'='*60}")
    print("  Ingestion complete. Run the scoring engine to verify results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpha Vantage financial ingestion")
    parser.add_argument(
        "--ev-only",
        action="store_true",
        help="Only (re)insert enterprise_value_mm — skips all other metric phases."
    )
    args = parser.parse_args()
    main(ev_only=args.ev_only)
