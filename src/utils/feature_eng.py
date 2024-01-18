import pandas as pd


def generate_features(input_data: pd.DataFrame) -> pd.DataFrame:

        # Convert pickup and dropoff cols to datetime
        input_data['tpep_pickup_datetime'] = pd.to_datetime(input_data['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
        input_data['tpep_dropoff_datetime'] = pd.to_datetime(input_data['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')
        #create month
        input_data['month'] = input_data['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
        # create day col
        input_data['day'] = input_data['tpep_pickup_datetime'].dt.day_name().str.lower()
        # create time of the day
        input_data['am_rush'] = input_data['tpep_pickup_datetime'].dt.hour
        input_data['day_time'] = input_data['tpep_pickup_datetime'].dt.hour
        input_data['pm_rush'] = input_data['tpep_pickup_datetime'].dt.hour
        input_data['night time'] = input_data['tpep_pickup_datetime'].dt.hour

        input_data['am_rush'] = input_data['am_rush'].apply(lambda x: 1 if 6 <= x < 10 else 0)
        input_data['day_time'] = input_data['am_rush'].apply(lambda x: 1 if 10 <= x < 16 else 0)
        input_data['pm_rush'] = input_data['am_rush'].apply(lambda x: 1 if 16<= x < 20 else 0)
        input_data['night_time'] = input_data['am_rush'].apply(lambda x : 1 if (20 <= x < 24) or (0 <= x < 6) else 0)

        # drop redundant columns
        drop_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        # convert catergorical features to string
        cols_to_str = ['RatecodeID', 'VendorID', 'DOLocationID', 'PULocationID']

        # Convert each column to string
        for col in cols_to_str:
            input_data[col] = input_data[col].astype('str')

        input_data = input_data.drop(columns=drop_cols, axis=1)

        return input_data
    