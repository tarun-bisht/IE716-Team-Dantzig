from src.scrape_cities_data import get_data_pd
import argparse

parser = argparse.ArgumentParser(description="Get cities data and distances between them from internet")
parser.add_argument("-n", "--names", required=True, type=str, help="Path to city names txt file")
parser.add_argument("-s", "--save_folder", type=str, help="Folder to save city data", default=None)

args = parser.parse_args()

city_df, distance_df = get_data_pd(args.names, args.save_folder)