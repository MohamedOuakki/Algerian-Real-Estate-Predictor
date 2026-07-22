import pandas as pd
import re
import os

def load_data(path="data/raw/listings.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "store_id", "ad_number", "ad_views"])
    return df


def extract_listing_type(title):
    title = str(title).lower()
    if "vente" in title:
        return "vente"
    elif "location" in title:
        return "location"
    elif "echange" in title:
        return "echange"
    else:
        return "other"


def extract_property_type(title):
    title = str(title).lower()
    if "appartement" in title:
        return "appartement"
    elif "villa" in title:
        return "villa"
    elif "terrain" in title:
        return "terrain"
    elif "local" in title:
        return "local"
    elif "maison" in title:
        return "maison"
    elif "studio" in title:
        return "studio"
    else:
        return "other"


def extract_rooms(title):
    # looks for F1, F2, F3 ... F10 in the title
    match = re.search(r'f(\d+)', str(title).lower())
    if match:
        return int(match.group(1))
    return None


def clean_price(df):
    # remove extreme outliers — keep prices between 10k and 500M DA
    df = df[(df["product_price"] >= 10_000) & (df["product_price"] <= 500_000_000)]
    return df


FOREIGN = ["france", "espagne", "turquie", "irlande"]

def clean_address(df):
    # strip quotes, spaces, normalize to lowercase
    df["store_adress"] = (
        df["store_adress"]
        .str.strip()
        .str.lower()
        .str.replace('"', '', regex=False)
    )
    # remove foreign listings
    df = df[~df["store_adress"].isin(FOREIGN)]
    return df


def extract_publishing_year(df):
    df["publishing_date"] = pd.to_datetime(df["publishing_date"], format="%d-%m-%Y à %H:%M", errors="coerce")
    df["year"] = df["publishing_date"].dt.year
    return df


def preprocess(input_path="data/raw/listings.csv",
               output_path="data/processed/listings_clean.csv"):

    print("Loading data...")
    df = load_data(input_path)
    print(f"  → {len(df)} rows loaded")

    print("Extracting features from title...")
    df["listing_type"]   = df["product_title"].apply(extract_listing_type)
    df["property_type"]  = df["product_title"].apply(extract_property_type)
    df["num_rooms"]      = df["product_title"].apply(extract_rooms)

    print("Cleaning price...")
    before = len(df)
    df = clean_price(df)
    print(f"  → Removed {before - len(df)} outlier rows")

    print("Cleaning address...")
    df = clean_address(df)

    print("Extracting year from date...")
    df = extract_publishing_year(df)

    print("Dropping unnecessary columns...")
    df = df.drop(columns=["product_title", "publishing_date", "category"])

    print("Encoding categorical columns...")
    df = pd.get_dummies(df, columns=["listing_type", "property_type", "store_adress"])

    # drop rows where num_rooms is null — we'll keep it as a feature with fill
    df["num_rooms"] = df["num_rooms"].fillna(0).astype(int)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDone. Clean dataset saved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    return df
    

if __name__ == "__main__":
    preprocess()