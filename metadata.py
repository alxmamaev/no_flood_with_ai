target_station_ids = [6005, 6022, 6027, 5004, 5012, 5024, 5805]

meteo_ids = [4443141, 5053701, 4971271, 4853511, 5313301, 5031331,
             4943011, 4961191, 5522471, 5152811, 5233581, 5171761,
             5402291, 5442741, 5032761, 5313601, 5293891, 5163331,
             5352391, 4783561, 4443211, 4982991, 4553551, 5302871,
             4433241, 4943321, 4653531, 4991581, 4483421, 4423391,
             5011811, 5253091, 5211881, 5013361, 5092851, 5233401, 4473211]

stantion_coord = {6005: {'lat': 53.48085555555556, 'lon': 123.88833055555556},
                 6022: {'lat': 50.25600277777778, 'lon': 127.51667},
                 6296: {'lat': 50.27, 'lon': 127.6},
                 6027: {'lat': 49.3, 'lon': 129.68},
                 5004: {'lat': 47.93, 'lon': 132.48},
                 5012: {'lat': 48.46, 'lon': 135.08},
                 5024: {'lat': 50.53, 'lon': 137.05},
                 5805: {'lat': 53.12, 'lon': 140.72}}

nearest_stantions = {6005: [(6246, 93.8), (6545, 989.0)],
                    6022: [(6020, 60.2), (6549, 140.1), (6016, 156.4)],
                    6296: [(6022, 6.1), (6023, 17.4), (6020, 60.5)],
                    6027: [(6507, 55.7), (6026, 82.0), (6443, 92.7)],
                    5004: [(5675, 22.3), (5044, 39.9), (5002, 73.8)],
                    5012: [(5013, 4.3), (5009, 34.0), (5008, 41.3)],
                    5024: [(5415, 33.3), (5420, 35.9), (5663, 55.7)],
                    5805: [(5803, 58.1), (5664, 79.2), (5033, 84.1)]}

nearest_meteo = {6005: [(4961191, 0.0),
                  (4553551, 98.5028731419143),
                  (4473211, 189.90302750703472),
                  (5313301, 198.0235001044391),
                  (4982991, 223.49405991406644)],
                6022: [(5013361, 0.0),
                  (5313601, 6.124422370955657),
                  (4653531, 16.786387326966594),
                  (4783561, 30.487124084549556),
                  (4971271, 36.586471285876925)],
                6296: [(5313601, 0.0),
                  (5013361, 6.12442237095566),
                  (4653531, 17.355839207780466),
                  (4783561, 27.656210883284356),
                  (4971271, 30.4984322967421)],
                6027: [(5302871, 0.0),
                  (5211881, 35.814971425979735),
                  (5011811, 52.34291918528278),
                  (4423391, 55.714426041325105),
                  (4483421, 55.97536089726729),
                  (5352391, 35.814971425979735)],
                5004: [(4991581, 0.0),
                  (5442741, 22.251445241687634),
                  (5032761, 39.85810571928348),
                  (5253091, 73.79081463752097),
                  (5031331, 90.97411607470794)],
                5012: [(4443141, 33.96562690635686),
                  (4853511, 41.31234203838251),
                  (5092851, 52.209448842210556),
                  (5402291, 60.001124303802705),
                  (5171761, 69.9562480054037)],
                5024: [(5152811, 0.0),
                  (5522471, 33.30465058840214),
                  (5293891, 76.4319141025116),
                  (5053701, 76.53867721831865),
                  (4433241, 91.1242866963961),
                  (5233401, 217.08126380925515)],
                5805: [(4443211, 0.0),
                  (5163331, 79.18979004157526),
                  (5233581, 84.12444091914531),
                  (4943321, 158.83781738828642),
                  (4943011, 281.5263015267437)]}


using_features = ['trend-3-level',
 'trend-180-level',
 'identifier',
 'trend-3-near_level_2',
 'trend-7-level',
 'month',
 'trend-3-near_level_3',
 'trend-30-mean-60-temperature_air',
 'trend-60-level',
 'trend-60-mean-60-temperature_air',
 'shift-3-trend-60-level',
 'shift-60-trend-180-level',
 'shift-30-trend-180-level',
 'shift-3-trend-3-level',
 'trend-11-level',
 'near_dist_2',
 'trend-15-level',
 'shift-180-mean-180-pressure',
 'trend-30-level',
 'shift-3-trend-15-level',
 'shift-17-trend-180-level',
 'shift-14-trend-180-level',
 'shift-30-mean-180-temperature_air',
 'shift-3-trend-7-level',
 'near_dist_3',
 'near_dist_1',
 'shift-60-mean-180-pressure',
 'shift-30-mean-60-temperature_air',
 'shift-3-trend-180-level',
 'temperature_air',
 'shift-7-trend-60-level',
 'shift-60-sum-180-precipitation_amount',
 'shift-60-trend-60-level',
 'shift-11-trend-60-near_level_1',
 'shift-60-mean-180-temperature_air',
 'trend-180-mean-180-temperature_air',
 'shift-11-trend-180-level',
 'shift-30-sum-180-precipitation_amount',
 'trend-3-near_level_1',
 'pressure',
 'shift-5-trend-60-level',
 'trend-60-mean-180-temperature_air',
 'trend-180-pressure',
 'shift-17-trend-60-level',
 'trend-60-mean-14-temperature_air',
 'trend-15-mean-60-temperature_air',
 'shift-3-trend-30-level',
 'shift-14-trend-60-near_level_1',
 'mean-14-temperature_air',
 'trend-7-sum-60-shower',
 'trend-11-mean-180-temperature_air',
 'shift-360-sum-180-humidity',
 'shift-7-trend-180-level',
 'trend-30-mean-180-pressure',
 'shift-180-mean-180-temperature_air',
 'trend-7-sum-30-shower',
 'trend-180-mean-60-temperature_air',
 'trend-7-near_level_3',
 'shift-5-trend-15-level',
 'trend-3-sum-180-rain',
 'shift-180-sum-30-precipitation_amount',
 'shift-180-mean-60-temperature_air',
 'trend-7-mean-180-temperature_air',
 'shift-180-sum-30-snow',
 'trend-180-mean-14-temperature_air',
 'sum-14-humidity',
 'trend-60-sum-60-humidity',
 'shift-180-mean-60-pressure',
 'trend-11-near_level_2',
 'trend-15-mean-180-temperature_air',
 'shift-17-trend-30-level',
 'trend-11-mean-60-temperature_air',
 'trend-60-near_level_1',
 'trend-3-sum-30-precipitation_amount',
 'trend-180-mean-14-pressure',
 'shift-14-trend-30-level',
 'mean-60-pressure',
 'trend-7-sum-180-rain',
 'trend-180-sum-60-humidity',
 'trend-7-mean-180-pressure',
 'shift-360-sum-180-precipitation_amount',
 'shift-30-trend-60-level',
 'trend-3-sum-60-humidity',
 'shift-5-trend-180-level',
 'trend-15-sum-60-shower',
 'shift-3-trend-11-level',
 'trend-180-mean-180-wind_speed',
 'trend-30-near_level_1',
 'shift-60-mean-14-temperature_air',
 'trend-30-mean-180-temperature_air',
 'shift-180-sum-14-precipitation_amount',
 'mean-3-temperature_air/ground',
 'shift-5-trend-180-temperature_air',
 'trend-11-sum-60-shower',
 'trend-180-near_level_1',
 'trend-11-sum-180-rain',
 'shift-360-mean-180-pressure',
 'shift-7-trend-15-level',
 'shift-30-mean-180-pressure',
 'shift-60-mean-60-pressure',
 'shift-30-trend-180-near_level_2',
 'trend-3-sum-60-precipitation_amount',
 'trend-60-sum-180-humidity',
 'trend-7-sum-14-shower',
 'shift-14-trend-60-level',
 'sum-30-humidity',
 'trend-60-mean-60-wind_direction',
 'shift-11-trend-11-level',
 'shift-30-mean-60-pressure',
 'trend-11-near_level_1',
 'trend-180-mean-3-pressure',
 'trend-15-mean-180-pressure',
 'trend-7-sum-180-shower',
 'trend-60-mean-180-temperature_ground',
 'trend-7-near_level_2',
 'shift-30-mean-14-temperature_air',
 'shift-30-sum-180-shower',
 'humidity',
 'shift-180-sum-180-precipitation_amount',
 'shift-180-mean-14-pressure',
 'shift-360-mean-14-temperature_air',
 'mean-180-wind_speed',
 'rain',
 'shift-17-trend-11-level',
 'trend-30-mean-180-visibility_distance',
 'precipitation_amount',
 'trend-180-mean-3-temperature_air',
 'trend-11-mean-180-pressure',
 'trend-15-near_level_2',
 'shift-360-mean-14-pressure',
 'trend-30-mean-180-temperature_ground',
 'mean-3-pressure',
 'shift-60-sum-180-humidity',
 'trend-180-sum-14-rain',
 'shift-180-mean-60-wind_speed',
 'trend-11-sum-60-humidity',
 'shift-17-trend-15-level',
 'shift-3-trend-3-near_level_2',
 'trend-11-mean-60-wind_direction',
 'shift-17-trend-60-near_level_1',
 'trend-11-near_level_3',
 'shift-14-trend-7-level',
 'mean-180-pressure',
 'trend-180-mean-60-temperature_ground',
 'trend-3-mean-180-pressure',
 'trend-30-sum-60-humidity',
 'trend-3-sum-180-precipitation_amount',
 'shift-30-mean-180-wind_speed',
 'shift-180-sum-60-snow',
 'trend-180-sum-14-shower',
 'trend-15-sum-180-rain',
 'trend-15-near_level_1',
 'trend-15-sum-60-humidity',
 'shift-180-sum-60-precipitation_amount',
 'shift-60-mean-180-temperature_ground',
 'trend-11-mean-180-temperature_ground',
 'shift-3-trend-60-near_level_3',
 'shift-30-trend-180-temperature_ground',
 'shift-30-trend-11-level',
 'shift-60-mean-14-visibility_distance',
 'trend-180-mean-60-pressure',
 'mean-60-wind_speed',
 'shift-11-trend-30-level',
 'trend-15-mean-14-temperature_air/ground',
 'trend-60-mean-60-temperature_ground',
 'shift-7-trend-60-near_level_1',
 'trend-60-sum-14-humidity',
 'trend-180-temperature_air',
 'shift-360-sum-14-humidity',
 'trend-3-sum-14-humidity',
 'trend-11-humidity',
 'shift-17-trend-60-near_level_2',
 'trend-180-mean-14-wind_direction',
 'sum-180-shower',
 'shift-180-mean-14-temperature_ground',
 'trend-180-sum-60-snow',
 'mean-14-wind_direction',
 'trend-11-sum-180-shower',
 'shift-60-mean-3-temperature_air',
 'trend-15-mean-180-temperature_ground',
 'shift-60-sum-180-fog',
 'shift-17-trend-180-temperature_ground',
 'trend-180-mean-3-temperature_ground',
 'shift-7-trend-30-level',
 'trend-60-mean-60-pressure',
 'trend-60-mean-3-temperature_air',
 'shift-180-mean-14-temperature_air',
 'trend-30-sum-14-humidity',
 'mean-14-pressure',
 'trend-30-sum-30-humidity',
 'shift-14-trend-180-temperature_ground',
 'trend-3-sum-30-shower',
 'trend-7-sum-180-precipitation_amount',
 'shift-180-mean-180-wind_speed',
 'shift-360-sum-30-humidity',
 'trend-11-sum-30-humidity',
 'shift-180-sum-180-rain',
 'trend-180-sum-14-fog',
 'trend-180-mean-60-visibility_distance',
 'trend-15-mean-14-wind_direction',
 'trend-60-sum-14-shower',
 'shift-60-sum-30-snow',
 'trend-3-mean-180-temperature_air',
 'shift-30-sum-60-snow',
 'shift-5-trend-11-level',
 'trend-15-sum-30-humidity',
 'shift-360-sum-30-snow',
 'shift-60-mean-60-wind_direction',
 'shift-180-sum-180-fog',
 'shift-30-sum-30-precipitation_amount',
 'shift-3-trend-180-temperature_ground',
 'trend-60-sum-180-snow',
 'shift-180-temperature_ground',
 'shift-180-mean-14-visibility_distance',
 'trend-60-near_level_2',
 'shift-30-sum-60-precipitation_amount',
 'trend-11-sum-180-fog',
 'sum-14-shower',
 'sum-30-fog',
 'shift-17-trend-180-temperature_air',
 'trend-30-mean-3-pressure',
 'shift-60-sum-60-fog',
 'shift-180-sum-180-humidity',
 'shift-30-trend-15-level',
 'shift-60-sum-14-humidity',
 'trend-15-mean-180-wind_direction',
 'trend-15-humidity',
 'mean-60-temperature_ground',
 'shift-30-sum-60-shower',
 'trend-3-mean-14-pressure',
 'trend-30-sum-60-fog',
 'mean-180-temperature_ground',
 'trend-180-sum-30-humidity',
 'shift-360-mean-60-pressure',
 'trend-15-mean-3-pressure',
 'shift-60-mean-14-wind_speed',
 'shift-14-trend-11-level',
 'trend-180-mean-60-temperature_air/ground',
 'shift-180-mean-180-temperature_ground',
 'trend-7-sum-14-humidity',
 'shift-5-trend-30-level',
 'trend-30-mean-60-visibility_distance',
 'trend-180-sum-60-precipitation_amount',
 'shift-360-mean-180-temperature_air',
 'trend-11-sum-30-shower',
 'trend-180-mean-180-temperature_ground',
 'trend-60-mean-180-visibility_distance',
 'trend-7-sum-180-fog',
 'trend-180-sum-30-precipitation_amount',
 'trend-180-mean-14-temperature_ground',
 'snow']