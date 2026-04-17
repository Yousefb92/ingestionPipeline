import os
import time
import json
import psycopg2
from datetime import date
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

"""
Pulls transcripts of earning calls for companies that are still active
Captures
Speaker 
Date
Content
Vectorised Content for RAG
Metadata
"""

# ─── SECRETS & SETUP ─────────────────────────────────────────────
load_dotenv()
ALPHA_VANTAGE_API_KEY = ""
GEMINI_API_KEY = ""

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ─── CONFIGURATION ───────────────────────────────────────────────
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Charizard006!",
    "host": "34.39.17.58",
    "port": "5432"
}

TICKERS = ["EMR", "NATI", "KEYS", "TER", "AME"]
START_YEAR = 2019
END_YEAR = 2022
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


# ─────────────────────────────────────────────────────────────────

def get_quarter_end_date(year, quarter_string):
    """Converts Q1, Q2, etc., into a specific date object for Postgres."""
    if quarter_string == "Q1": return date(year, 3, 31)
    if quarter_string == "Q2": return date(year, 6, 30)
    if quarter_string == "Q3": return date(year, 9, 30)
    if quarter_string == "Q4": return date(year, 12, 31)
    return date(year, 12, 31)


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
        print(f"  -> [!] Embedding failed: {e}")
        return None


def run_ingestion():
    print("Starting End-to-End Transcript Ingestion...\n" + "=" * 50)

    # 1. Connect to the database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    # 2. Fetch company IDs (It is faster to cache them once than query 60 times)
    cursor.execute("SELECT ticker, id FROM companies;")
    company_map = {row[0]: row[1] for row in cursor.fetchall()}

    # 3. Main Fetch Loop
    for ticker in TICKERS:

        # Verify the ticker exists in your database
        company_id = company_map.get(ticker)
        if not company_id:
            print(f"[-] Ticker {ticker} not found in 'companies' table. Skipping.")
            continue

        for year in range(START_YEAR, END_YEAR + 1):
            for q in QUARTERS:

                # Stop pulling data for Q3 or Q4 of 2022
                if year == 2022 and q in ["Q3", "Q4"]:
                    print('Guardrail in place to not pull data for ',year,q)
                    continue

                quarter_param = f"{year}{q}"
                print(f"\nProcessing {ticker} {quarter_param}...")

                # Fetch from Alpha Vantage
                url = f'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={quarter_param}&apikey={ALPHA_VANTAGE_API_KEY}'
                print(url)
                response = requests.get(url)

                if response.status_code != 200:
                    print(f"  -> [!] HTTP Error: {response.status_code}")
                    continue

                data = response.json()

                if "transcript" not in data or not data["transcript"]:
                    print(f"  -> [-] No transcript found.")
                    continue

                source_date = get_quarter_end_date(year, q)
                inserted_count = 0

                # Parse the transcript blocks
                for block in data['transcript']:
                    if block.get('title') == 'Operator':
                        continue

                    content = block.get('content', '')
                    if not content.strip():
                        continue

                    # Generate the 1536-dim vector via Gemini
                    vector = generate_embedding(content)
                    if not vector:
                        continue

                    # Bundle the metadata
                    metadata = {
                        "source_type": "earnings_call",
                        "quarter": quarter_param,
                        "speaker": block.get('speaker'),
                        "title": block.get('title'),
                        "sentiment": float(block.get('sentiment', 0.0))
                    }

                    # Insert into Postgres
                    cursor.execute("""
                        INSERT INTO strategic_context
                        (company_id, content, metadata, source_date, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        company_id,
                        content,
                        json.dumps(metadata),
                        source_date,
                        vector
                    ))
                    inserted_count += 1

                # Commit the transaction for this specific quarter
                conn.commit()
                print(f"  -> [+] Embedded and saved {inserted_count} speaking blocks.")

                # Respect Alpha Vantage free tier rate limits (5 per minute)
                time.sleep(1)

    cursor.close()
    conn.close()
    print("\n" + "=" * 50 + "\nIngestion Complete!")


if __name__ == "__main__":
    run_ingestion()