import pandas as pd
import pycountry


def country_to_iso3(name):
    if pd.isna(name):
        return pd.NA

    name = str(name).strip()

    country = pycountry.countries.get(name=name)
    if country is not None:
        return country.alpha_3

    try:
        matches = pycountry.countries.search_fuzzy(name)
        return matches[0].alpha_3
    except LookupError:
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
    "Congo DR": "COD",
    "Macau": "MAC",
    "U.S. Virgin Islands": "VIR",
}

def build_holiday_features():

    df = pd.read_csv("data/raw/holidays_raw.csv")

    df["iso3"] = df["Country"].apply(country_to_iso3)
    df["iso3"] = df.apply(
        lambda row: manual_fix.get(row["Country"], row["iso3"]),
        axis=1
    )


    #convert date format
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )


    #cleaning
    df = df.rename(columns={
        "Holiday": "holiday_name"
    })

    df = df[[
        "iso3",
        "date",
        "year",
        "month",
        "holiday_name"
    ]]

    df["holiday_name"] = (
        df["holiday_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["holiday_name"] = df["holiday_name"].replace({

         "labour day": "labor day",
        "eid al-adha holiday": "eid al-adha",
        "eid al-fitr holiday": "eid al-fitr",
        "new year": "new year's day",

        # nowruz variants
        "nooruz national holiday": "nowruz",
        "nowruz bayram": "nowruz",
        "nowruz/spring holiday": "nowruz",
        "nevruz day": "nowruz",
        "navruz celebration": "nowruz",
        "'nooruz national holiday' observed": "nowruz",
        "nauruz": "nowruz",
        "nowruz holiday": "nowruz",
        "'nevruz day' day off": "nowruz",
        "nooruz holiday": "nowruz",
        "'nowruz/spring holiday' day off": "nowruz",

        # eid al-fitr variants
        "eid ul fitr": "eid al-fitr",
        "eid ul-fitr": "eid al-fitr",
        "eid-ul-fitr": "eid al-fitr",
        "eid el fitr": "eid al-fitr",
        "eid el-fitr": "eid al-fitr",
        "eid el fitri": "eid al-fitr",
        "eid ul-fitr holiday": "eid al-fitr",
        "eid el fitr holiday": "eid al-fitr",
        "eid-ul-fitr holiday": "eid al-fitr",
        "eid-ul-fithr": "eid al-fitr",
        "eid-ul-fithr holiday": "eid al-fitr",
        "'eid al-fitr' observed": "eid al-fitr",
        "'eid al-fitr' day off": "eid al-fitr",
        "'eid al-fitr holiday' observed": "eid al-fitr",
        "'eid al-fitr holiday' day off": "eid al-fitr",
        "eid al-fitr eve": "eid al-fitr",
        "holiday for eid al-fitr": "eid al-fitr",
        "eid al-fitr public sector holiday": "eid al-fitr",

        # eid al-adha variants
        "eid ul adha holiday": "eid al-adha",
        "eid-ul al'haa": "eid al-adha",
        "eid-ul al'haa holiday": "eid al-adha",
        "eid al-adha holiday": "eid al-adha",
        "eid al-adha eve": "eid al-adha",
        "eid al-adha (feast of sacrifice)": "eid al-adha",
        "eid al-adha (for muslims only)": "eid al-adha",
        "'eid al-adha' observed": "eid al-adha",
        "'eid al-adha' day off": "eid al-adha",
        "'eid al-adha holiday' day off": "eid al-adha",
        "eid al-adha observed": "eid al-adha",
        "eid al adha public sector holiday": "eid al-adha",
        "eid al-adha public sector holiday": "eid al-adha",
        "eid al-adha day 2": "eid al-adha",
        "eid al-adha day 3": "eid al-adha",

        # prophet birthday variants
        "eid e-milad-un nabi": "mawlid",
        "eid milad un-nabi": "mawlid",

        # sacrifice variants
        "eid al-qurban": "eid al-adha",
        "eid al-qurban holiday": "eid al-adha",

    })





    df["holiday_religion_detail"] = "none"

    # islam
    df.loc[df["holiday_name"].str.contains("eid al-fitr|eid ul fitr|ramadan", na=False), "holiday_religion_detail"] = "shared_islam"
    df.loc[df["holiday_name"].str.contains("eid al-adha", na=False), "holiday_religion_detail"] = "shared_islam"
    df.loc[df["holiday_name"].str.contains("mawlid|prophet's birthday|prophets birthday", na=False), "holiday_religion_detail"] = "shared_islam"
    df.loc[df["holiday_name"].str.contains("ashura", na=False), "holiday_religion_detail"] = "shia_islam"
    df.loc[df["holiday_name"].str.contains("eid-e-ghadir|eid al-ghadeer", na=False), "holiday_religion_detail"] = "shia_islam"
    df.loc[df["holiday_name"].str.contains("prophet's birthday|muharram",na=False), "holiday_religion_detail"] = "shared_islam"
    df.loc[df["holiday_name"].str.contains("ashura", na=False), "holiday_religion_detail"] = "shia_islam"
    df.loc[df["holiday_name"].str.contains("prophet muhammad",na=False), "holiday_religion_detail"] = "shared_islam"

    # christianity
    df.loc[df["holiday_name"].str.contains("christmas|good friday|easter|ascension|pentecost|whit monday|all saints|assumption", na=False), "holiday_religion_detail"] = "shared_christianity"
    df.loc[df["holiday_name"].str.contains("epiphany|corpus christi|holy saturday|maundy thursday|whit sunday|immaculate conception|all souls",na=False), "holiday_religion_detail"] = "shared_christianity"
    df.loc[df["holiday_name"].str.contains("orthodox good friday|reformation day|carnival tuesday",na=False), "holiday_religion_detail"] = "shared_christianity"

    # other religions / cultural
    df.loc[df["holiday_name"].str.contains("diwali|holi", na=False), "holiday_religion_detail"] = "hinduism"
    df.loc[df["holiday_name"].str.contains("buddha|vesak", na=False), "holiday_religion_detail"] = "buddhism"
    df.loc[df["holiday_name"].str.contains("hanukkah|yom kippur|passover", na=False), "holiday_religion_detail"] = "judaism"
    df.loc[df["holiday_name"].str.contains("nowruz|navruz|nevruz", na=False), "holiday_religion_detail"] = "cultural"
    df.loc[df["holiday_name"].str.contains("maha shivaratri",na=False), "holiday_religion_detail"] = "hinduism"



    #monthly counts per religion group
    holiday_monthly = (
        df.groupby(["iso3", "date", "year", "month", "holiday_religion_detail"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    holiday_monthly = holiday_monthly.rename(columns={
        "shared_christianity": "christian_holiday_count",
        "shared_islam": "islam_holiday_count",
        "shia_islam": "shia_holiday_count",
        "hinduism": "hindu_holiday_count",
        "buddhism": "buddhist_holiday_count",
        "judaism": "jewish_holiday_count",
        "cultural": "cultural_holiday_count",
        "none": "nonreligious_holiday_count"
    })

    # full country-month grid
    all_months = pd.date_range(
        start=holiday_monthly["date"].min(),
        end=holiday_monthly["date"].max(),
        freq="MS"
    )

    all_countries = holiday_monthly["iso3"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_countries, all_months],
        names=["iso3", "date"]
    )

    full_df = pd.DataFrame(index=full_index).reset_index()

    holiday_full = full_df.merge(
        holiday_monthly,
        on=["iso3", "date"],
        how="left"
    )

    # fill missing counts
    count_cols = [c for c in holiday_full.columns if "holiday_count" in c]
    holiday_full[count_cols] = holiday_full[count_cols].fillna(0)

    # restore year/month
    holiday_full["year"] = holiday_full["date"].dt.year
    holiday_full["month"] = holiday_full["date"].dt.month

    count_cols = [c for c in holiday_full.columns if "holiday_count" in c]
    holiday_full['holiday_count_month'] = holiday_full[count_cols].sum(axis=1)
    holiday_full['is_holiday_month'] = (holiday_full['holiday_count_month'] > 0).astype(int)

    return holiday_full

def add_holiday_features(combined, gdf=None):

    holiday_full = build_holiday_features()

    count_cols = [c for c in holiday_full.columns if "holiday_count" in c]
    flag_cols  = [c for c in ["is_holiday_month"] if c in holiday_full.columns]
    merge_cols = count_cols + flag_cols

    combined_reset = combined.reset_index()
    rename_map = {}

    if "iso3" not in combined_reset.columns:
        if "matched_admin1_id" in combined_reset.columns:
           combined_reset["iso3"] = combined_reset["matched_admin1_id"].str.split(" - ").str[0].str.strip().str.upper()
        else:
            raise KeyError(f"Cannot find iso3 column. Available: {combined_reset.columns.tolist()}")

    if "date" not in combined_reset.columns:
        if "month_year" in combined_reset.columns:
            rename_map["month_year"] = "date"
        else:
            raise KeyError(f"Cannot find date column. Available: {combined_reset.columns.tolist()}")

    combined_reset = combined_reset.rename(columns=rename_map)

    combined_reset["date"] = pd.to_datetime(combined_reset["date"])
    holiday_full["date"] = pd.to_datetime(holiday_full["date"])

    combined_reset["date"] = combined_reset["date"].dt.to_period("M").dt.to_timestamp()
    holiday_full["date"] = holiday_full["date"].dt.to_period("M").dt.to_timestamp()

    combined_reset = combined_reset.merge(
        holiday_full[["iso3", "date"] + merge_cols],
        on=["iso3", "date"],
        how="left"
    )

    combined_reset[merge_cols] = combined_reset[merge_cols].fillna(0)
    reverse_map = {v: k for k, v in rename_map.items()}
    combined_reset = combined_reset.rename(columns=reverse_map)

    combined = combined_reset.set_index(list(combined.index.names))

    return combined