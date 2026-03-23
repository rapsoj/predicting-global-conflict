import os
import pandas as pd
import geopandas as gpd
from utils import data_cleaning, map_admin_regions
from config import settings
from utils.features.holidays import add_holiday_features
from utils.features.worldbank import add_worldbank_features
from utils.features.religion import add_religion_features


#def prepare_data_pipeline(clean_data: bool = False):
#    """
#    Builds or loads the model-ready DataFrame.
#    If clean_data=True, load from saved file. Otherwise, run the full pipeline.
 #   """
  #  output_path = "data/processed/model_data.csv"

   # if not clean_data and os.path.exists(output_path):
    #    print("Loading cleaned data from disk...")
     #   df = pd.read_csv(output_path, index_col=[0, 1])
      #  return df
#
 #   print("Running full data preprocessing pipeline...")
  #  df = pd.read_csv("data/raw/1997-01-01-2025-07-03.csv")
   # df = df[df['year'] >= 2018].copy()
    #df['date'] = pd.to_datetime(df['event_date'], format='%d %B %Y')
    #df['month_year'] = df['date'].dt.to_period('M').dt.to_timestamp()

    #gdf = gpd.read_file("data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
    
    #if 'country_code' not in df.columns:
     #   df['country_code'] = df['country']

# admin1 column naming fix
    #if 'admin1' not in df.columns:
     #   if 'admin1_name' in df.columns:
      #      df['admin1'] = df['admin1_name']
       # elif 'admin1_region' in df.columns:
        #    df['admin1'] = df['admin1_region']

   # df_neighbours = map_admin_regions.add_admin1_neighbors(df, gdf)
     

    #neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)
    #event_data = data_cleaning.get_monthly_events(df_neighbours)
    #subevent_data = data_cleaning.get_monthly_subevents(
     #   df_neighbours, ['Excessive force against protesters', 'Agreement']
    #)

    #combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    #combined = data_cleaning.add_lagged_columns(combined)
    #combined = data_cleaning.add_time_trend_features(combined)
    #combined = data_cleaning.add_importance_weights(combined)
    #combined = add_worldbank_features(combined, gdf)
    #combined = add_holiday_features(combined, gdf)
    #combined = add_religion_features(combined)
    


    #model_data = combined[settings.predictors + settings.targets]

    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #model_data.to_csv(output_path)

    #return model_data
    
def prepare_data_pipeline(clean_data: bool = False):
    output_path = "data/processed/model_data.csv"

    if not clean_data and os.path.exists(output_path):
        print("Loading cleaned data from disk...")
        df = pd.read_csv(output_path) # Don't set index yet to avoid prefix issues
        return df

    print("Running full data preprocessing pipeline...")
    df = pd.read_csv("data/raw/1997-01-01-2025-07-03.csv")
    df = df[df['year'] >= 2018].copy()
    
    # 1. Standardize Date immediately
    df['date'] = pd.to_datetime(df['event_date'], format='%d %B %Y')
    df['month_year'] = df['date'].dt.to_period('M').dt.to_timestamp()

    gdf = gpd.read_file("data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
    
    # 2. Geography Mapping
    df_neighbours, gdf_updated = map_admin_regions.add_admin1_neighbors(df, gdf)

    # 3. Create Base Monthly Aggregates
    event_data = data_cleaning.get_monthly_events(df_neighbours)
    subevent_data = data_cleaning.get_monthly_subevents(
        df_neighbours, ['Excessive force against protesters', 'Agreement']
    )
    neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)

    # 4. Join and Standardize Index for Features
    combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    
    # CRITICAL: Ensure month_year is Datetime before adding features
    combined = combined.reset_index()
    combined['month_year'] = pd.to_datetime(combined['month_year'])
    combined = combined.set_index(['matched_admin1_id', 'month_year'])

    # 5. Add Features (One by One)
    combined = data_cleaning.add_lagged_columns(combined)
    combined = data_cleaning.add_time_trend_features(combined)
    combined = data_cleaning.add_importance_weights(combined)
    
    # These functions need the 'gdf' to map ISO codes correctly
    combined = add_worldbank_features(combined, gdf_updated)
    combined = add_holiday_features(combined, gdf_updated)
    combined = add_religion_features(combined)

    # 6. Final Clean up
    # Fill event NaNs with 0, but leave WB/Religion NaNs if you want to see if they worked
    event_cols = [c for c in combined.columns if '(t-1)' in c or c in settings.targets]
    combined[event_cols] = combined[event_cols].fillna(0)

    model_data = combined[settings.predictors + settings.targets]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_data.to_csv(output_path)

    return model_data


def filter_admin1_data(df, admin1_region):
    return df.loc[admin1_region]