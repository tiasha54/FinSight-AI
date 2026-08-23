"""
clean_finance_data.py
---------------------
Cleans and preprocesses Personal_Finance_Dataset_1.csv for RAG training.

"""

import pandas as pd
import numpy as np
import re
import os

# ─────────────────────────────────────────────
# PATH SETUP — works on Windows, Mac, Linux
# Always reads/writes from the folder this script is in
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "Personal_Finance_Dataset_1.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "cleaned_finance_data.csv")


# ─────────────────────────────────────────────
# STEP 1: Load
# ─────────────────────────────────────────────
print("=" * 50)
print("  AI Finance Analyzer — Data Cleaner")
print("=" * 50)

print("\n[1/10] Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"       Rows loaded: {len(df)}")
print(f"       Columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 2: Strip whitespace from all text columns
# ─────────────────────────────────────────────
print("\n[2/10] Stripping whitespace...")
text_cols = ["Transaction Description", "Category", "Type"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
print("       Done")


# ─────────────────────────────────────────────
# STEP 3: Parse dates + extract time features
# ─────────────────────────────────────────────
print("\n[3/10] Parsing dates and extracting time features...")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")

bad_dates = df["Date"].isna().sum()
if bad_dates > 0:
    print(f"       WARNING: Dropped {bad_dates} rows with unparseable dates")
    df = df.dropna(subset=["Date"])

df["Year"]       = df["Date"].dt.year
df["Month"]      = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%B")
df["Day"]        = df["Date"].dt.day
df["Weekday"]    = df["Date"].dt.strftime("%A")
df["Quarter"]    = df["Date"].dt.quarter

df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
print("       New columns added: Year, Month, Month_Name, Day, Weekday, Quarter")


# ─────────────────────────────────────────────
# STEP 4: Clean Transaction Description
# ─────────────────────────────────────────────
print("\n[4/10] Cleaning transaction descriptions...")

def clean_description(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s\&\-\/\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown transaction"

df["Transaction Description"] = df["Transaction Description"].apply(clean_description)
print("       Done")


# ─────────────────────────────────────────────
# STEP 5: Standardise Category and Type casing
# ─────────────────────────────────────────────
print("\n[5/10] Standardising Category and Type...")
df["Category"] = df["Category"].str.strip().str.title()
df["Type"]     = df["Type"].str.strip().str.capitalize()
print("       Done")


# ─────────────────────────────────────────────
# STEP 6: Validate Amount
# ─────────────────────────────────────────────
print("\n[6/10] Validating Amount column...")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

bad_amount = df["Amount"].isna() | (df["Amount"] <= 0)
if bad_amount.sum() > 0:
    print(f"       WARNING: Dropped {bad_amount.sum()} rows with invalid Amount")
    df = df[~bad_amount]

df["Amount"] = df["Amount"].round(2)
print(f"       Amount range: {df['Amount'].min()} to {df['Amount'].max()}")


# ─────────────────────────────────────────────
# STEP 7: Remove duplicates
# ─────────────────────────────────────────────
print("\n[7/10] Removing duplicates...")
before = len(df)
df = df.drop_duplicates()
print(f"       Removed {before - len(df)} duplicate rows")


# ─────────────────────────────────────────────
# STEP 8: Drop remaining nulls
# ─────────────────────────────────────────────
print("\n[8/10] Dropping remaining null rows...")
before = len(df)
df = df.dropna()
print(f"       Dropped {before - len(df)} rows")


# ─────────────────────────────────────────────
# STEP 9: Encode columns for ML
# ─────────────────────────────────────────────
print("\n[9/10] Encoding Type and Category for ML...")
df["Is_Income"] = (df["Type"] == "Income").astype(int)

categories = sorted(df["Category"].unique())
cat_map = {cat: idx for idx, cat in enumerate(categories)}
df["Category_Code"] = df["Category"].map(cat_map)

print("       Category encoding map:")
for name, code in cat_map.items():
    print(f"          {code:>2} -> {name}")


# ─────────────────────────────────────────────
# STEP 10: Shuffle + reorder columns + save
# ─────────────────────────────────────────────
print("\n[10/10] Shuffling and saving...")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

final_columns = [
    "Date", "Year", "Month", "Month_Name", "Day", "Weekday", "Quarter",
    "Transaction Description",
    "Category", "Category_Code",
    "Amount",
    "Type", "Is_Income"
]
df = df[final_columns]

df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("  CLEANING COMPLETE")
print("=" * 50)
print(f"\n  Output file : cleaned_finance_data.csv")
print(f"  Saved at    : {OUTPUT_FILE}")
print(f"  Final shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n  Category distribution:")
print(df["Category"].value_counts().to_string())
print(f"\n  Type distribution:")
print(df["Type"].value_counts().to_string())
print(f"\n  Preview (first 3 rows):")
print(df.head(3).to_string())
print("\n  File is ready for RAG training!")