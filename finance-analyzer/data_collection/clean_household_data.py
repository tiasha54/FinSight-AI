"""
clean_household_data.py
-----------------------
Cleans and preprocesses Daily_Household_Transactions_2.csv for RAG training.

"""

import pandas as pd
import numpy as np
import re
import os

# PATH SETUP
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "Daily_Household_Transactions_2.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "cleaned_household_data.csv")


print("=" * 55)
print("  AI Finance Analyzer — Household Data Cleaner")
print("=" * 55)


# STEP 1: Load
print("\n[1/11] Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"       Rows loaded  : {len(df)}")
print(f"       Columns      : {list(df.columns)}")


# STEP 2: Strip whitespace
print("\n[2/11] Stripping whitespace from text columns...")
text_cols = ["Date", "Mode", "Category", "Subcategory", "Note", "Income/Expense", "Currency"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
print("       Done")


# STEP 3: Parse dates (mixed formats)
print("\n[3/11] Parsing dates (mixed formats)...")

def parse_mixed_date(date_str):
    date_str = str(date_str).strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return pd.NaT

df["Date_Parsed"] = df["Date"].apply(parse_mixed_date)

bad_dates = df["Date_Parsed"].isna().sum()
if bad_dates > 0:
    print(f"       WARNING: Dropped {bad_dates} rows with unparseable dates")
    df = df.dropna(subset=["Date_Parsed"])

df["Date"]       = df["Date_Parsed"].dt.strftime("%Y-%m-%d")
df["Year"]       = df["Date_Parsed"].dt.year
df["Month"]      = df["Date_Parsed"].dt.month
df["Month_Name"] = df["Date_Parsed"].dt.strftime("%B")
df["Day"]        = df["Date_Parsed"].dt.day
df["Weekday"]    = df["Date_Parsed"].dt.strftime("%A")
df["Quarter"]    = df["Date_Parsed"].dt.quarter
df = df.drop(columns=["Date_Parsed"])

print(f"       Date range: {df['Date'].min()} to {df['Date'].max()}")
print("       New columns: Year, Month, Month_Name, Day, Weekday, Quarter")


# STEP 4: Fill nulls in Subcategory and Note
print("\n[4/11] Filling nulls in Subcategory and Note...")
df["Subcategory"] = df["Subcategory"].replace("nan", "Unknown").fillna("Unknown")
df["Note"]        = df["Note"].replace("nan", "Unknown").fillna("Unknown")
print("       Done")


# STEP 5: Standardise Category
print("\n[5/11] Standardising Category...")

def clean_category(cat):
    cat = str(cat).strip()
    mapping = {
        "subscription"         : "Subscription",
        "maid"                 : "Maid",
        "scrap"                : "Scrap",
        "garbage disposal"     : "Garbage Disposal",
        "water (jar /tanker)"  : "Water",
        "Small Cap fund 2"     : "Small Cap Fund 2",
        "Small cap fund 1"     : "Small Cap Fund 1",
    }
    return mapping.get(cat, cat.title())

df["Category"] = df["Category"].apply(clean_category)
print(f"       Unique categories after cleaning: {df['Category'].nunique()}")


# STEP 6: Standardise Mode
print("\n[6/11] Standardising Mode...")
df["Mode"] = df["Mode"].str.strip().str.title()
print(f"       Unique modes: {df['Mode'].nunique()}")


# STEP 7: Standardise Transaction Type
print("\n[7/11] Standardising Income/Expense column...")
df["Income/Expense"] = df["Income/Expense"].str.strip().str.title()
df = df.rename(columns={"Income/Expense": "Transaction_Type"})
print(f"       Unique types: {df['Transaction_Type'].unique()}")


# STEP 8: Drop Currency column (all INR)
print("\n[8/11] Dropping Currency column (all values are INR)...")
df = df.drop(columns=["Currency"])
print("       Done")


# STEP 9: Validate Amount
print("\n[9/11] Validating Amount...")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

bad_amount = df["Amount"].isna() | (df["Amount"] <= 0)
if bad_amount.sum() > 0:
    print(f"       WARNING: Dropped {bad_amount.sum()} rows with invalid Amount")
    df = df[~bad_amount]

df["Amount"] = df["Amount"].round(2)
print(f"       Amount range: {df['Amount'].min()} to {df['Amount'].max()}")


# STEP 10: Encode for ML
print("\n[10/11] Encoding Transaction_Type and Category for ML...")

type_map = {"Expense": 0, "Income": 1, "Transfer-Out": 2}
df["Transaction_Type_Code"] = df["Transaction_Type"].map(type_map).fillna(-1).astype(int)

categories = sorted(df["Category"].unique())
cat_map = {cat: idx for idx, cat in enumerate(categories)}
df["Category_Code"] = df["Category"].map(cat_map)

print("       Transaction Type: Expense=0, Income=1, Transfer-Out=2")
print(f"       Total categories encoded: {len(cat_map)}")


# STEP 11: Remove duplicates, shuffle, reorder, save
print("\n[11/11] Removing duplicates, shuffling and saving...")

before = len(df)
df = df.drop_duplicates()
print(f"       Removed {before - len(df)} duplicate rows")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

final_columns = [
    "Date", "Year", "Month", "Month_Name", "Day", "Weekday", "Quarter",
    "Mode",
    "Category", "Category_Code",
    "Subcategory",
    "Note",
    "Amount",
    "Transaction_Type", "Transaction_Type_Code"
]
df = df[final_columns]

df.to_csv(OUTPUT_FILE, index=False)


# FINAL SUMMARY
print("\n" + "=" * 55)
print("  CLEANING COMPLETE")
print("=" * 55)
print(f"\n  Output file : cleaned_household_data.csv")
print(f"  Saved at    : {OUTPUT_FILE}")
print(f"  Final shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n  Transaction Type distribution:")
print(df["Transaction_Type"].value_counts().to_string())
print(f"\n  Top 10 Categories:")
print(df["Category"].value_counts().head(10).to_string())
print(f"\n  Preview (first 3 rows):")
print(df.head(3).to_string())
print("\n  File is ready for RAG training!")