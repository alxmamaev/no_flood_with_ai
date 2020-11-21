import pandas as pd
import numpy as np
from functools import partial
import multiprocessing
from tqdm import tqdm
import os
from metadata import using_features, target_station_ids

def shift(data, columns, timedelta):
    shifted_values = data[columns].shift(timedelta)
    shifted_values.columns = ["shift-{}-{}".format(timedelta, i) for i in shifted_values.columns]
    
    return shifted_values


def generate_shifted_features(data, feature_columns, timedeltas):
    shifted_features = []
    
    for timedelta in timedeltas:
        shifted_features.append(shift(data, feature_columns, timedelta))
        
    shifted_features = pd.concat(shifted_features, axis=1)

    return shifted_features


def trend(data, columns, timedelta):
    trend_values = data[columns].diff(timedelta)
    trend_values.columns = ["trend-{}-{}".format(timedelta, i) for i in trend_values.columns]
    
    return trend_values


def generate_trend_features(data, feature_columns, timedeltas):
    trend_features = []
    
    for timedelta in timedeltas:
        trend_features.append(trend(data, feature_columns, timedelta))
        
    trend_features = pd.concat(trend_features, axis=1)

    return trend_features


def aggregate(data, timedelta_days, columns, agg_function, time):
    end_time = time
    start_time = end_time - pd.Timedelta(days=(timedelta_days - 1))
    interval_data = data[start_time:end_time][columns]
    
    if interval_data.shape[0] < 1:
        return pd.Series(dtype=np.float32)
    
    agg_data = getattr(interval_data, agg_function)().round(1)
    agg_data.index = ["{}-{}-{}".format(agg_function, timedelta_days, i) for i in agg_data.index]
    
    return agg_data


def generate_aggregation_features(data, feature_columns, timedeltas, agg_function_name="mean"):
    aggregation_features = []
    
    timeline = data.reset_index().time
    for timedelta_days in timedeltas:
        agg_function = partial(aggregate, data, timedelta_days, feature_columns, agg_function_name)
        aggregation_features.append(timeline.apply(agg_function))
        
    aggregation_features = pd.concat(aggregation_features, axis=1)
    aggregation_features.index = data.index
            
    return aggregation_features


def process_stantion_data(stantion_data):
    temperature_ground = stantion_data["temperature_ground"].copy()
    temperature_ground[temperature_ground == 0.0] = 1
    
    stantion_data["temperature_air/ground"] = stantion_data["temperature_air"] / temperature_ground
    stantion_data["is_snow"] = ((stantion_data["snow"] > 0) & (stantion_data["shower"] > 0)).astype(np.float32)
    stantion_data["is_rain"] = (stantion_data["rain"] > 0).astype(np.float32)
    
    
    trend_features_1 = generate_trend_features(stantion_data,
                                             ['level', 'near_level_1', 'near_level_2', 'near_level_3', 
                                               'visibility_distance', 'humidity', 'temperature_air',
                                               'wind_speed', 'temperature_ground', 
                                              'pressure', 'temperature_air/ground'],
                                             [3, 7, 11, 15, 30, 60, 180])
    
    shift_features_1 = generate_shifted_features(stantion_data,
                                                ["humidity", "temperature_air", 
                                                 "wind_speed", "wind_direction", 
                                                 "temperature_ground", "pressure",
                                                 "temperature_air/ground"],
                                                  [3, 7, 30, 60, 180, 365])
    
    shift_features_2 = generate_shifted_features(trend_features_1,
                                                 trend_features_1.columns.tolist(),
                                                 [3, 5, 7, 11, 14, 17, 30, 60])
    
    
    agg_features_1 = generate_aggregation_features(stantion_data,
                                                  ['visibility_distance', 'temperature_air',
                                                   'wind_speed', 'wind_direction', 
                                                   'temperature_ground', 'pressure',
                                                   'temperature_air/ground'],
                                                   [3, 14, 60, 180])
    
    agg_features_2 = generate_aggregation_features(stantion_data,
                                                   ["precipitation_amount", "humidity", 
                                                    "rain", "shower", "snow", "fog", "drizzle"],
                                                   [14, 30, 60, 180],
                                                   agg_function_name="sum")  
    
    trend_features_2 = generate_trend_features(agg_features_1,
                                               agg_features_1.columns.tolist(),
                                               [3, 7, 11, 15, 30, 60, 180])
    
    trend_features_3 = generate_trend_features(agg_features_2,
                                               agg_features_2.columns.tolist(),
                                               [3, 7, 11, 15, 30, 60, 180])
    
    
    shift_features_3 = generate_shifted_features(agg_features_1,
                                               agg_features_1.columns.tolist(),
                                               [30, 60, 180, 360])
    
    shift_features_4 = generate_shifted_features(agg_features_2,
                                               agg_features_2.columns.tolist(),
                                               [30, 60, 180, 360])
    
    
    weather = generate_aggregation_features(stantion_data,
                                           ["is_snow", "is_rain", "humidity", "temperature_air",
                                            "precipitation_amount", "pressure", "wind_direction",
                                            "wind_speed"],
                                           [10],
                                           agg_function_name="mean") 
    
    weather = generate_shifted_features(weather,
                                        weather.columns.tolist(),
                                        [-10])
    
    weather.columns = ["snow", "rain", "humidity", "temperature_air",
                        "precipitation_amount", "pressure", "wind_direction",
                        "wind_speed"]
    
    
    X = pd.concat((trend_features_1, trend_features_2, trend_features_3, 
                   shift_features_1, shift_features_2, shift_features_3, shift_features_4,
                   agg_features_1, agg_features_2,
                   weather), axis=1)
    
    
    stantion_id = stantion_data["identifier"].iloc[0]
    X["identifier"] = stantion_id
    X["month"] = stantion_data.index.month
    X["day"] = stantion_data.index.day
    
    X["near_dist_1"] = stantion_data["near_dist_1"]
    X["near_dist_2"] = stantion_data["near_dist_2"]
    X["near_dist_3"] = stantion_data["near_dist_3"]
    
    X = X[using_features]
    level = stantion_data["level"]


    nan_indexes = X.isna().any(axis=1)
    drop_indexed = X.index[nan_indexes]
    
    X = X.drop(drop_indexed)
    level = level.drop(drop_indexed)
    
    return stantion_id, (X, level)


def process_data(data):
    splited_data = []
    for stantion_id in target_station_ids:
        stantion_data = data.loc[stantion_id]
        stantion_data["identifier"] = stantion_id
        
        splited_data.append(stantion_data)
        
    pool = multiprocessing.Pool(os.cpu_count())
    # data_processor = (process_stantion_data(i) for i in splited_data)
    data_processor = pool.imap(process_stantion_data, splited_data)
    data_processor = tqdm(data_processor, total=len(splited_data))
    
    processed_data = dict(data_processor)
    
    pool.close()
    return processed_data