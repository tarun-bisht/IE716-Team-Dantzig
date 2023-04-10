import requests
import pandas as pd
import math
import os
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")


def get_cities_data(city: str):
    data = requests.get(f"https://nominatim.openstreetmap.org/search?q={city}&format=json").json()[0]
    loc_id = data["osm_id"]
    lat = data["lat"]
    lon = data["lon"]
    name = data["display_name"]
    try:
        data = {"id": loc_id, "name":city, "lat": float(lat), "lon": float(lon), "location": name}
    except ValueError as e:
        print(e)
        data = None
    return data

def calculate_distance(city1: dict, city2: dict):
    lon1 = math.radians(city1["lon"])
    lon2 = math.radians(city2["lon"])
    lat1 = math.radians(city1["lat"])
    lat2 = math.radians(city2["lat"])
    # Haversine formula
    lon_diff = lon2 - lon1
    lat_diff = lat2 - lat1
    a = math.sin(lat_diff / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(lon_diff / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth in kilometers.
    r = 6357
    # calculate the result
    return c * r

def get_data_pd(city_names_file: str, save_folder_path: str = None):
    with open(city_names_file, "r") as data:
        cities = [city.strip() for city in data.readlines()]
        
    cities_data = []
    city_df = pd.DataFrame(columns=["id", "name", "lat", "lon", "location"])
    print("Downloading Cities Data ...")
    for city in tqdm(cities):
        data = get_cities_data(city)
        city_df = city_df.append(data,ignore_index=True)
        cities_data.append(data)
    if save_folder_path is not None:
        city_df.to_csv(os.path.join(save_folder_path, "cities_data.csv"), index=True)
    
    distance_df = pd.DataFrame(columns=cities, index=cities)
    print("Calculating Distance Matrix ...")
    for city1 in tqdm(cities_data):
        for city2 in cities_data:
            distance_df.loc[city1["name"], city2["name"]] = calculate_distance(city1, city2)
    if save_folder_path is not None:
        distance_df.to_csv(os.path.join(save_folder_path, "cities_distances.csv"), index=True)
    return city_df, distance_df


if __name__ == "__main__":

    with open("city.names", "r") as data:
        cities = [city.strip() for city in data.readlines()]
        
    cities_data = []
    city_df = pd.DataFrame(columns=["id", "name", "lat", "lon", "location"])
    for city in cities:
        data = get_cities_data(city)
        city_df = city_df.append(data,ignore_index=True)
        cities_data.append(data)
    city_df.to_csv("cities_data.csv", index=True)
    
    distance_df = pd.DataFrame(columns=cities, index=cities)
    for city1 in cities_data:
        for city2 in cities_data:
            distance_df.loc[city1["name"], city2["name"]] = calculate_distance(city1, city2)
    distance_df.to_csv("cities_distances.csv", index=True)
