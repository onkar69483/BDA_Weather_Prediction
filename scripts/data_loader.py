import os
import zipfile
import pandas as pd
from pymongo import MongoClient

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ZIP_PATH = os.path.join(DATA_DIR, "storm_events.zip")

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "weather_db"
COLLECTION_NAME = "storm_events"


# -----------------------------
# 2. EXTRACT ZIP
# -----------------------------
def extract_dataset():
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"❌ ZIP file not found: {ZIP_PATH}")

    print(f"📦 Extracting dataset from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("❌ No CSV files found after extraction.")

    print(f"✅ Found {len(csv_files)} CSV file(s).")
    return [os.path.join(DATA_DIR, f) for f in csv_files]


# -----------------------------
# 3. LOAD & CLEAN DATA
# -----------------------------
def load_and_clean(csv_files):
    print("🧹 Loading and cleaning CSV files...")
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    if not dfs:
        raise ValueError("❌ No valid CSVs could be read.")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Combined dataset shape before cleaning: {combined_df.shape}")

    # Drop completely empty columns
    combined_df.dropna(axis=1, how="all", inplace=True)
    # Drop rows missing critical info like event type or date
    combined_df.dropna(subset=["EVENT_TYPE", "BEGIN_DATE_TIME"], inplace=True)

    # Keep only first 50,000 records
    limited_df = combined_df.head(50000).reset_index(drop=True)

    print(f"✅ Final dataset shape (limited to 50k): {limited_df.shape}")
    return limited_df


# -----------------------------
# 4. LOAD INTO MONGODB
# -----------------------------
def load_to_mongo(df):
    print("💾 Loading data into MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Clean previous records (optional)
    collection.delete_many({})
    records = df.to_dict("records")

    collection.insert_many(records)
    print(f"✅ Successfully inserted {len(records)} records into {DB_NAME}.{COLLECTION_NAME}")

    client.close()


# -----------------------------
# 5. MAIN FUNCTION
# -----------------------------
if __name__ == "__main__":
    csv_files = extract_dataset()
    df = load_and_clean(csv_files)
    load_to_mongo(df)