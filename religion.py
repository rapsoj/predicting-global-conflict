import pandas as pd
import pycountry


def country_to_iso3(name):
    if pd.isna(name):
        return pd.NA
    name = str(name).strip()
    country = pycountry.countries.get(name=name)
    if country:
        return country.alpha_3
    try:
        result = pycountry.countries.search_fuzzy(name)
        return result[0].alpha_3
    except:
        return pd.NA


manual_fix = {
    "St. Lucia": "LCA",
    "St. Vincent and the Grenadines": "VCT",
    "Antigua & Barbuda": "ATG",
    "St. Kitts and Nevis": "KNA",
    "Cape Verde": "CPV",
    "Ivory Coast": "CIV",
    "Democratic Republic of the Congo": "COD",
    "Swaziland": "SWZ",
    "Turkey": "TUR",
    "East Timor": "TLS",
}


def build_religion_features():
    df = pd.read_csv("data/raw/WRP_national.csv")
    cow = pd.read_csv("data/raw/COW-country-codes.csv")

    cow = cow[["CCode", "StateNme"]].drop_duplicates()
    df = df.merge(cow, left_on="state", right_on="CCode", how="left")

    df["iso3"] = df["StateNme"].apply(country_to_iso3)
    df["iso3"] = df.apply(
        lambda row: manual_fix.get(str(row["StateNme"]), row["iso3"]), axis=1
    )

    df = df[df["iso3"].notna()].copy()
    df = df[df["year"] == 2010].copy()

    pct_cols = [c for c in df.columns if c.endswith("pct") and c not in [
        "chrstgenpct", "judgenpct", "islmgenpct", "budgenpct", "sumreligpct"
    ]]

    def get_top_religions(row):
        vals = row[pct_cols].fillna(0).sort_values(ascending=False).head(3)
        row["majority_religion"] = vals.index[0]
        row["majority_pct"] = vals.iloc[0]
        row["minority1_religion"] = vals.index[1]
        row["minority1_pct"] = vals.iloc[1]
        row["minority2_religion"] = vals.index[2]
        row["minority2_pct"] = vals.iloc[2]
        return row

    df = df.apply(get_top_religions, axis=1)

    return df[[
        "iso3",
        "majority_religion", "majority_pct",
        "minority1_religion", "minority1_pct",
        "minority2_religion", "minority2_pct",
        "nonreligpct"
    ]]


def add_religion_features(combined, gdf=None):
    religion_df = build_religion_features()

    combined_reset = combined.reset_index()

    combined_reset["iso3"] = combined_reset["matched_admin1_id"].str.extract(r'([A-Z]{3})')

    combined_reset = combined_reset.merge(
        religion_df,
        on="iso3",
        how="left"
    )

    combined_reset = combined_reset.drop(columns=["iso3"])
    combined = combined_reset.set_index(["matched_admin1_id", "month_year"])

    return combined

