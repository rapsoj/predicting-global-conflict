import argparse
from utils.preprocessing import prepare_data_pipeline, filter_admin1_data
from models.simple_model import train_and_evaluate_model

def forecast_admin1_events(target_admin1: str, target_event: str, clean_data: bool = False):
    """
    Full modeling pipeline for a given ADMIN1 region and target event type.
    If clean_data=True, skips processing and loads from saved file.
    """
    model_data = prepare_data_pipeline(clean_data=clean_data)
    region_data = filter_admin1_data(model_data, target_admin1)
    train_and_evaluate_model(region_data, target_event, region_name=target_admin1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast conflict events at the Admin1 level.")
    parser.add_argument("--region", type=str, required=True, help="Target ADMIN1 region name")
    parser.add_argument("--event", type=str, required=True, help="Target event type (e.g., Battles)")
    parser.add_argument("--clean-data", action="store_true", help="Run full data cleaning pipeline")

    args = parser.parse_args()

    forecast_admin1_events(
        target_admin1=args.region,
        target_event=args.event,
        clean_data=args.clean_data
    )