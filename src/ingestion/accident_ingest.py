"""
Ingestion script for Traffic Accident records.
Maintains a raw landing copy, records audit logs, and handles faulty records.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import uuid
import logging
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import text
from src.etl.db import get_engine, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("accident_ingest")


def generate_benchmark_accidents_dataset(num_records: int = 5000, inject_faults: bool = True) -> pd.DataFrame:
    """
    Generates a realistic benchmark traffic accidents dataset conforming to
    UK Road Safety Open Data standards (with coordinates, road types, severities).
    Optionally injects a small fraction of faulty records to test validation and quarantine.
    """
    np.random.seed(42)
    
    # Representative UK urban & highway clusters (London, Birmingham, Manchester, Leeds, Glasgow)
    cities = [
        {"name": "London", "lat": 51.5074, "lon": -0.1278, "authority": "Greater London Authority"},
        {"name": "Birmingham", "lat": 52.4862, "lon": -1.8904, "authority": "Birmingham City Council"},
        {"name": "Manchester", "lat": 53.4808, "lon": -2.2426, "authority": "Manchester City Council"},
        {"name": "Leeds", "lat": 53.8008, "lon": -1.5491, "authority": "Leeds City Council"},
        {"name": "Glasgow", "lat": 55.8642, "lon": -4.2518, "authority": "Glasgow City Council"}
    ]
    
    city_choices = np.random.choice(cities, size=num_records)
    lats = [c["lat"] + np.random.normal(0, 0.05) for c in city_choices]
    lons = [c["lon"] + np.random.normal(0, 0.05) for c in city_choices]
    authorities = [c["authority"] for c in city_choices]
    
    # Realistic dates across the past 2 years
    base_date = datetime.date(2024, 1, 1)
    day_offsets = np.random.randint(0, 730, size=num_records)
    dates = [base_date + datetime.timedelta(days=int(d)) for d in day_offsets]
    
    raw_hour_p = np.array([
        0.015, 0.01, 0.008, 0.007, 0.01, 0.02, 0.05, 0.08, 0.09, 0.06,  # 00-09
        0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.09, 0.09, 0.07, 0.05,     # 10-19
        0.04, 0.03, 0.02, 0.02                                           # 20-23
    ])
    hour_p = raw_hour_p / raw_hour_p.sum()
    hours = np.random.choice(range(24), size=num_records, p=hour_p)
    minutes = np.random.randint(0, 60, size=num_records)
    times = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]
    
    # Severities: Severe class imbalance as seen in real world
    # 1 = Fatal (~2%), 2 = Serious (~18%), 3 = Slight (~80%)
    severity_codes = np.random.choice([1, 2, 3], size=num_records, p=[0.025, 0.185, 0.79])
    
    road_types = np.random.choice(
        ["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"],
        size=num_records,
        p=[0.70, 0.18, 0.06, 0.04, 0.02]
    )
    
    speed_limits = np.random.choice([20, 30, 40, 50, 60, 70], size=num_records, p=[0.10, 0.60, 0.10, 0.05, 0.08, 0.07])
    
    light_conditions = np.random.choice(
        ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"],
        size=num_records,
        p=[0.72, 0.20, 0.03, 0.05]
    )
    
    road_surfaces = np.random.choice(
        ["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood over 3cm"],
        size=num_records,
        p=[0.68, 0.26, 0.04, 0.015, 0.005]
    )
    
    weather_conds = np.random.choice(
        ["Fine no high winds", "Raining no high winds", "Raining + high winds", "Fog or mist", "Snowing"],
        size=num_records,
        p=[0.75, 0.16, 0.04, 0.03, 0.02]
    )
    
    urban_rural = np.random.choice(["Urban", "Rural"], size=num_records, p=[0.65, 0.35])
    vehicles = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.35, 0.55, 0.08, 0.02])
    casualties = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.78, 0.16, 0.04, 0.02])
    
    indices = [f"ACC_{2024}_{i:06d}" for i in range(1, num_records + 1)]
    
    df = pd.DataFrame({
        "accident_index": indices,
        "accident_date": [d.strftime("%Y-%m-%d") for d in dates],
        "accident_time": times,
        "latitude": np.round(lats, 6),
        "longitude": np.round(lons, 6),
        "accident_severity": severity_codes,
        "number_of_vehicles": vehicles,
        "number_of_casualties": casualties,
        "speed_limit": speed_limits,
        "road_type": road_types,
        "light_conditions": light_conditions,
        "weather_conditions": weather_conds,
        "road_surface_conditions": road_surfaces,
        "urban_or_rural_area": urban_rural,
        "local_authority": authorities
    })
    
    # Inject deliberate faults into ~1.5% of records if enabled to verify quarantine pipeline
    if inject_faults:
        fault_count = max(5, int(num_records * 0.015))
        fault_indices = np.random.choice(num_records, size=fault_count, replace=False)
        for i, idx in enumerate(fault_indices):
            if i % 3 == 0:
                # Invalid latitude
                df.at[idx, "latitude"] = 195.0
            elif i % 3 == 1:
                # Invalid speed limit
                df.at[idx, "speed_limit"] = -10
            else:
                # Missing date
                df.at[idx, "accident_date"] = None
                
    return df


def ingest_accidents(num_records: int = 5000, raw_output_dir: str = "data/raw") -> str:
    """
    Main ingestion execution function:
    1. Acquires accident dataset.
    2. Writes immutable raw copy in data/raw/.
    3. Loads raw batch into stg_accidents_raw.
    4. Logs audit entry in ingestion_audit.
    """
    batch_id = f"batch_acc_{uuid.uuid4().hex[:8]}"
    raw_dir = Path(raw_output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    raw_file_path = raw_dir / f"accidents_raw_{batch_id}.csv"
    logger.info(f"Generating benchmark accident dataset (batch: {batch_id})...")
    
    df = generate_benchmark_accidents_dataset(num_records=num_records, inject_faults=True)
    df["batch_id"] = batch_id
    
    # 1. Save immutable raw landing copy
    df.to_csv(raw_file_path, index=False)
    logger.info(f"Saved raw landing file to: {raw_file_path} ({len(df)} records)")
    
    # 2. Database staging load & audit logging
    engine = get_engine()
    init_db(engine)
    
    start_time = datetime.datetime.now(datetime.timezone.utc)
    try:
        # Load directly to staging table
        df.to_sql("stg_accidents_raw", con=engine, if_exists="append", index=False)
        
        # Log successful ingestion audit
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ingestion_audit (batch_id, source_name, file_or_endpoint, status, records_ingested, records_rejected, started_at, completed_at)
                    VALUES (:batch_id, :source, :filepath, 'SUCCESS', :records, 0, :started, :completed)
                """),
                {
                    "batch_id": batch_id,
                    "source": "UK Road Safety Open Data Benchmark",
                    "filepath": str(raw_file_path),
                    "records": len(df),
                    "started": start_time,
                    "completed": datetime.datetime.now(datetime.timezone.utc)
                }
            )
        logger.info(f"Ingestion batch {batch_id} logged to audit successfully.")
        return batch_id
        
    except Exception as e:
        logger.error(f"Error staging accident batch {batch_id}: {e}")
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ingestion_audit (batch_id, source_name, file_or_endpoint, status, records_ingested, records_rejected, started_at, completed_at, error_message)
                    VALUES (:batch_id, :source, :filepath, 'FAILED', 0, 0, :started, :completed, :err)
                """),
                {
                    "batch_id": batch_id,
                    "source": "UK Road Safety Open Data Benchmark",
                    "filepath": str(raw_file_path),
                    "started": start_time,
                    "completed": datetime.datetime.now(datetime.timezone.utc),
                    "err": str(e)
                }
            )
        raise e


if __name__ == "__main__":
    ingest_accidents(num_records=2500)
