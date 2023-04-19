# from random import randint
# import pandas as pd
# from faker import Faker

# fake = Faker()

# def input_data(x):

#     # pandas dataframe
#     data = pd.DataFrame()
#     for i in range(0, x):
#         data.loc[i,'id']= int(randint(1, 15))
#         data.loc[i,'days']= int(randint(1, 10))
#     return data


# df = input_data(300)
# df.to_csv('data.csv')

import numpy as np
import pandas as pd
import random

# Define the number of samples
n_samples = 1000

# Generate random latitude and longitude values
latitudes = np.random.uniform(low=-90.0, high=90.0, size=n_samples)
longitudes = np.random.uniform(low=-180.0, high=180.0, size=n_samples)

# Generate random request dates
start_date = np.datetime64("2022-01-01")
end_date = np.datetime64("2022-12-31")
timedelta = end_date - start_date
request_dates = np.random.randint(0, timedelta.astype(
    'timedelta64[D]').astype(int), size=n_samples)
request_dates = start_date + request_dates

# Generate random days between requests
days_between_requests = np.random.randint(1, 10, size=n_samples)

# Combine the data into a Pandas dataframe
data = {
    "Latitude": latitudes,
    "Longitude": longitudes,
    "Request Date": request_dates,
    "Days Between Requests": days_between_requests
}

request_data = pd.DataFrame(data)

# Save the synthetic data to a CSV file
request_data.to_csv("synthetic_request_data.csv", index=False)
