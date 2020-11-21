import os
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

tqdm.pandas()

def convert_visiability(item):
    value = item["horizontalVisibility"]
    quality = item["horizontalVisibilityQuality"]

    if value == 99 and quality == 8:
        return np.nan

    if value == 0:
        return 0.03

    if value <= 50:
        return value / 10
    
    if 56 <= value <= 80:
        return value - 50.0
    
    if 81 <= value <= 88:
        return 35.0 + (value - 81) * 5
    
    if value == 89:
        return 75.0
    
    if value == 90:
        return 0.01
    
    if value == 91:
        return 0.05
    
    if value == 92:
        return 0.2
    
    if value == 93:
        return 0.5
    
    if value == 94:
        return 1.0
    
    if value == 95:
        return 2.0
    
    if value == 96:
        return 4.0
    
    if value == 97:
        return 10.0
    
    if value == 98:
        return 20.0
    
    if value == 98:
        return 55
    
    return np.nan

def convert_weather(item):
    result = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], 
                        index=["fog", "drizzle", "rain", "snow", "shower"],
                        dtype=np.float32)


    present_weather_matching_dict = {
        "fog": [11, 12, 28] + list(range(40, 50)),
        "drizzle": [20, 24, 68, 69] + list(range(50, 60)),
        "rain": [21, 23, 24, 79, 91, 92, 27, 58, 59, 93, 94, 95, 97] + list(range(60, 70)),
        "snow": [22, 23, 26, 27, 68, 69, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 97] + list(range(70, 80)),
        "shower": [25, 26, ] + list(range(80, 90))
    }

    past_weather_matching_dict = {
        "fog": [4],
        "drizzle": [5],
        "rain": [6],
        "snow": [7],
        "shower": [8]
    }

    if not np.isnan(item["pastWeather"]):
        for weather_key, weather_ids in past_weather_matching_dict.items():
            if int(item["pastWeather"]) in weather_ids:
                result.loc[weather_key] += 1

    if not np.isnan(item["presentWeather"]):
        for weather_key, weather_ids in present_weather_matching_dict.items():
            if int(item["presentWeather"]) in weather_ids:
                result.loc[weather_key] += 1

    return result
     
def _load_meteo_new(meteo_id, start_date=None, end_date=None):
    meteo = pd.read_csv("datasets/meteo_new/{}.csv".format(meteo_id), sep=",", index_col=0)

    meteo["year"] = meteo["localYear"]
    meteo["month"] = meteo["localMonth"]
    meteo["day"] = meteo["localDay"]
    
    meteo["date"] = pd.to_datetime(meteo[["year", "month", "day"]])

    if start_date is not None:
        meteo = meteo.set_index("date", drop=True)
        meteo = meteo.sort_index()
        meteo = meteo.loc[start_date:]
        meteo = meteo.reset_index()

    meteo["visiability"] = meteo.apply(convert_visiability, axis=1)
    weather = meteo.apply(convert_weather, axis=1)
    meteo = pd.concat((meteo, weather), axis=1)

    meteo["pastWeather"] = meteo["pastWeather"].fillna(-1).astype(np.int64)

    columns = ["date", "visiability",
               "relativeHumidity", "airTemperature", "windSpeed", "windDirection",
               "soilTemperature", "pressure", "totalAccumulatedPrecipitation",
               "fog", "drizzle", "rain", "snow", "shower"]

    meteo = meteo[columns]
    meteo = meteo.set_index("date", drop=True)
    meteo = meteo.sort_index()
    meteo = meteo.rename(columns={"relativeHumidity": "humidity",
                                "airTemperature": "temperature_air",
                                "visiability": "visibility_distance",
                                "soilTemperature": "temperature_ground",
                                "windDirection": "wind_direction",
                                "windSpeed": "wind_speed",
                                "totalAccumulatedPrecipitation": "precipitation_amount"})
    
    return meteo.resample("D").mean()

def load_meteo(meteo_id, start_date=None, end_date=None):
    try:
        return _load_meteo_new(meteo_id, start_date=start_date, end_date=end_date)
    except FileNotFoundError:
        pass

    meteo = pd.read_csv("datasets/meteo/4263131.csv", sep=";")
    meteo = meteo[['visibility_distance', 'wind_direction', 'wind_speed_avg', 
                'precipitation_amount', 'temperature_ground', 'temperature_air', 
                'humidity', 'time']]

    meteo = meteo.rename(columns={"wind_speed_avg": "wind_speed",
                                "time": "date"})
    meteo["date"] = pd.to_datetime(meteo["date"])
    meteo = meteo.set_index("date")
    if start_date is not None:
        meteo = meteo.loc()[start_date:]

    meteo = meteo.sort_index()
    meteo = meteo.resample("D").mean()
    return meteo


def calculate_meteo(meteo, nearest_meteo, item):
    identifier = item["identifier"]
    near_meteo = nearest_meteo[identifier]
    date = item["time"]
    
    meteo_dist = []
    meteo_data = []
    
    for i, dist in near_meteo:
        try:
            meteo_data.append(meteo[i].loc[date])
            meteo_dist.append(dist)
        except KeyError:
            continue
    
    if not meteo_data:
        return pd.Series(dtype=np.float32)

    meteo_dist_ = np.array(meteo_dist)
    meteo_dist_ = np.exp(-(meteo_dist_ / 10))
    
    meteo_data = pd.concat(meteo_data, axis=1)
    
    result = (meteo_data.fillna(0) * np.expand_dims(meteo_dist_, axis=0)).sum(axis=1)
    sum_dist = ((~meteo_data.isna()).astype(np.float32) * np.expand_dims(meteo_dist_, axis=0)).sum(axis=1)
    
    
    result = result / sum_dist
    result["min_stantion_dist"] = min(meteo_dist)
    
    meteo_dist_exp_sum_ = meteo_dist_.sum()
    result["mean_stantion_dist"] = sum([i * j / meteo_dist_exp_sum_ for i, j in zip(meteo_dist, meteo_dist_)]) 
    
    return result.round(2)