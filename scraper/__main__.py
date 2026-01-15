import os
import json
import time
import io
import requests
import pandas as pd
from datetime import timedelta
from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor, as_completed

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_LIST_FILE = os.path.join(PACKAGE_DIR, "token_list.jsonl")
PROCESSING_LOG = os.path.join(PACKAGE_DIR, "processing_log.jsonl")
OUTPUT_DIR = "scraped_data"
MIN_DATE = "2020-01-01"
MAX_DATE = "2024-12-31"
THREE_MONTHS_DAYS = 90
MAX_WORKERS = 1

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_malicious_eth_tokens():
    print("Fetching Malicious Tokens (Dianxiang-Sun Dataset)...")
    url = "https://raw.githubusercontent.com/dianxiang-sun/rug_pull_dataset/main/rugpull_full_dataset_new.csv"
    
    response = requests.get(url)
    response.raise_for_status()
    
    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
    df['Chain_Upper'] = df['Chain'].astype(str).str.upper()
    eth_tokens = df[df['Chain_Upper'].str.contains('ETH')]['address'].dropna().unique().tolist()
    
    print(f"  Found {len(eth_tokens)} malicious ETH tokens.")
    return eth_tokens

def get_normal_eth_tokens(client, limit=None):
    print(f"Querying BigQuery for Normal Tokens ({MIN_DATE} to {MAX_DATE})...")
    
    query = f"""
        SELECT address 
        FROM `bigquery-public-data.crypto_ethereum.tokens`
        WHERE block_timestamp >= TIMESTAMP('{MIN_DATE}')
          AND block_timestamp <= TIMESTAMP('{MAX_DATE}')
    """
    if limit:
        query += f" LIMIT {limit}"
    
    query_job = client.query(query)
    results = query_job.result()
    addrs = [row.address for row in results]
    print(f"  Found {len(addrs)} normal tokens via BigQuery.")
    return addrs

def fetch_and_save_token_list(client):
    ensure_dir("data")
    
    malicious_tokens = get_malicious_eth_tokens()
    malicious_set = set(malicious_tokens)
    
    raw_normal_tokens = get_normal_eth_tokens(client)
    normal_tokens = [addr for addr in raw_normal_tokens if addr not in malicious_set]
    print(f"Filtered Normal Tokens: {len(normal_tokens)} (after removing malicious overlaps)")
    
    new_data = []
    for addr in malicious_tokens:
        new_data.append({'address': addr, 'flag': 1})
    for addr in normal_tokens:
        new_data.append({'address': addr, 'flag': 0})
        
    new_df = pd.DataFrame(new_data)
    new_df.to_json(TOKEN_LIST_FILE, orient='records', lines=True)
    print(f"Saved {len(new_df)} tokens to {TOKEN_LIST_FILE}")

def get_processed_addresses():
    if not os.path.exists(PROCESSING_LOG):
        return set()
    
    processed = set()
    try:
        with open(PROCESSING_LOG, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'address' in entry:
                        processed.add(entry['address'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading log: {e}")
    
    return processed

def append_to_log(log_entry):
    try:
        with open(PROCESSING_LOG, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to log: {e}")

def process_single_token(data):
    addr = data.get('address')
    flag = data.get('flag')
    
    if not addr:
        return {'address': None, 'status': 'error', 'message': 'Missing address'}

    try:
        client = bigquery.Client(project="gen-lang-client-0129428353")
    except Exception as e:
        return {'address': addr, 'status': 'failed', 'message': f"Client Init Failed: {str(e)}"}

    filename = f"{OUTPUT_DIR}/{addr}_{flag}.csv"
    
    try:
        query = f"""
            WITH token_range AS (
                SELECT 
                    MIN(block_timestamp) as start_time, 
                    MAX(block_timestamp) as end_time
                FROM `bigquery-public-data.crypto_ethereum.token_transfers`
                WHERE token_address = '{addr}'
            )
            SELECT t.*
            FROM `bigquery-public-data.crypto_ethereum.token_transfers` t, token_range r
            WHERE t.token_address = '{addr}'
              AND TIMESTAMP_DIFF(r.end_time, r.start_time, DAY) >= {THREE_MONTHS_DAYS}
              AND t.block_timestamp <= TIMESTAMP_ADD(r.start_time, INTERVAL {THREE_MONTHS_DAYS} DAY)
            ORDER BY t.block_timestamp ASC
        """
        
        df = client.query(query).to_dataframe()
        
        if not df.empty:
            df.to_csv(filename, index=False)
            return {'address': addr, 'status': 'done', 'message': f"Saved {len(df)} rows"}
        else:
            return {'address': addr, 'status': 'skipped', 'message': 'No transactions or history < 90 days'}
            
    except Exception as e:
        return {'address': addr, 'status': 'failed', 'message': str(e)}

def process_all_tokens():
    ensure_dir(OUTPUT_DIR)
    
    if not os.path.exists(TOKEN_LIST_FILE):
        print(f"Token list {TOKEN_LIST_FILE} not found. Run with --fetch first.")
        return

    print("Loading token list...")
    try:
        token_df = pd.read_json(TOKEN_LIST_FILE, lines=True)
        all_tokens = token_df.to_dict('records')
    except ValueError:
        print("Token list is empty or invalid.")
        return

    processed_addresses = get_processed_addresses()
    tokens_to_process = [t for t in all_tokens if t['address'] not in processed_addresses]
    
    total = len(all_tokens)
    remaining = len(tokens_to_process)
    print(f"Total tokens: {total}. Already processed: {total - remaining}. Remaining: {remaining}")
    
    if remaining == 0:
        print("All tokens processed.")
        return

    print(f"Starting parallel processing with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_token = {executor.submit(process_single_token, token): token for token in tokens_to_process}
        
        completed_count = 0
        
        for future in as_completed(future_to_token):
            result = future.result()
            
            log_entry = {
                'address': result['address'],
                'status': result['status'],
                'timestamp': time.time(),
                'message': result.get('message', '')
            }
            append_to_log(log_entry)
            
            completed_count += 1
            
            status_symbol = "[+]" if result['status'] == 'done' else "[-]" if result['status'] == 'skipped' else "[!]"
            print(f"({completed_count}/{remaining}) {status_symbol} {result['address']} : {result['status']} - {result['message']}")

    print("Processing complete.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Token Scraping Script")
    parser.add_argument('--fetch', action='store_true', help='Fetch and save token list')
    parser.add_argument('--process', action='store_true', help='Process token transactions')
    parser.add_argument('--all', action='store_true', help='Run both fetch and process')
    
    args = parser.parse_args()
    
    if not (args.fetch or args.process or args.all):
        parser.print_help()
        return
    
    client = bigquery.Client()
    
    if args.fetch or args.all:
        print("=" * 60)
        print("STEP 1: Fetching Token Lists")
        print("=" * 60)
        fetch_and_save_token_list(client)
    
    if args.process or args.all:
        print("\n" + "=" * 60)
        print("STEP 2: Processing Token Transactions")
        print("=" * 60)
        process_all_tokens()

if __name__ == "__main__":
    main()
