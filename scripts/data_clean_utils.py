import numpy as np
import pandas as pd


columns_to_drop =  ["worker_id",
                    "company_latitude",
                    "company_longitude",
                    "service_latitude",
                    "service_longitude",
                    "request_date",
                    "requested_time_hour",
                    "city_name",
                    "request_day_of_week",
                    "request_month"]


def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "worker_age": "age",
            "worker_ratings": "ratings",
            "service_location_latitude": "service_latitude",
            "service_location_longitude": "service_longitude",
            "time_requested": "requested_time",
            "weather_conditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type"},
            axis=1)
    )


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    minors_data = data.loc[data["age"].astype("float") < 18]
    minor_index = minors_data.index.tolist()
    six_star_data = data.loc[data["ratings"] == "6"]
    six_star_index = six_star_data.index.tolist()

    return (
        data
        .drop(columns="request_id")
        .drop(index=minor_index)                                                # Minor workers in data dropped
        .drop(index=six_star_index)                                             # six star rated drivers dropped
        .replace("NaN ",np.nan)                                                 # missing values in the data
        .assign(
            # city column out of worker id
            city_name = lambda x: x["worker_id"].str.split("RES").str.get(0),
            # convert age to float
            age = lambda x: x["age"].astype(float),
            # convert ratings to float
            ratings = lambda x: x["ratings"].astype(float),
            # absolute values for location based columns
            company_latitude = lambda x: x["company_latitude"].abs(),
            company_longitude = lambda x: x["company_longitude"].abs(),
            service_latitude = lambda x: x["service_latitude"].abs(),
            service_longitude = lambda x: x["service_longitude"].abs(),
            # request date to datetime and feature extraction
            request_date = lambda x: pd.to_datetime(x["request_date"],
                                                  dayfirst=True),
            request_day = lambda x: x["request_date"].dt.day,
            request_month = lambda x: x["request_date"].dt.month,
            request_day_of_week = lambda x: x["request_date"].dt.day_name().str.lower(),
            is_weekend = lambda x: (x["request_date"]
                                    .dt.day_name()
                                    .isin(["Saturday","Sunday"])
                                    .astype(int)),
            # time based columns
            requested_time = lambda x: pd.to_datetime(x["requested_time"],
                                                  format="mixed"),
            worker_dispatch_time = lambda x: pd.to_datetime(x['worker_dispatch_time'],
                                                         format='mixed'),
            # time taken to pick request
            response_time_minutes = lambda x: (
                                            (x["worker_dispatch_time"] - x["requested_time"])
                                            .dt.seconds / 60
                                            ),
            # hour in which request was placed
            requested_time_hour = lambda x: x["requested_time"].dt.hour,
            # time of the day when request was placed
            requested_time_of_day = lambda x: (
                                x["requested_time_hour"].pipe(time_of_day)),
            # categorical columns
            weather = lambda x: (
                                x["weather"]
                                .str.replace("conditions ","")
                                .str.lower()
                                .replace("nan",np.nan)),
            traffic = lambda x: x["traffic"].str.rstrip().str.lower(),
            type_of_vehicle = lambda x: x["type_of_vehicle"].str.rstrip().str.lower(),
            festival = lambda x: x["festival"].str.rstrip().str.lower(),
            city_type = lambda x: x["city_type"].str.rstrip().str.lower(),
            # multiple request column
            multiple_requests = lambda x: x["multiple_requests"].astype(float),
            # target column modifications
            time_taken = lambda x: (x["time_taken"]
                                    .str.replace("(min) ","")
                                    .astype(int)))
        .drop(columns=["requested_time","worker_dispatch_time"])
    )
    
    
    
def clean_lat_long(data: pd.DataFrame, threshold: float=1.0) -> pd.DataFrame:
    location_columns = ["company_latitude",
                        "company_longitude",
                        "service_latitude",
                        "service_longitude"]

    return (
        data
        .assign(**{
            col: (
                np.where(data[col] < threshold, np.nan, data[col].values)
            )
            for col in location_columns
        })
    )
    
    
# extract day, day name, month and year
def extract_datetime_features(ser):
    date_col = pd.to_datetime(ser,dayfirst=True)

    return (
        pd.DataFrame(
            {
                "day": date_col.dt.day,
                "month": date_col.dt.month,
                "year": date_col.dt.year,
                "day_of_week": date_col.dt.day_name(),
                "is_weekend": date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
            }
        ))
    
    
def time_of_day(ser):

    return(
        pd.cut(ser,bins=[0,6,12,17,20,24],right=True,
               labels=["after_midnight","morning","afternoon","evening","night"])
    )


def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = data.drop(columns=columns)
    return df


def calculate_haversine_distance(df):
    location_columns = ["company_latitude",
                        "company_longitude",
                        "service_latitude",
                        "service_longitude"]
    
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )

def create_distance_type(data: pd.DataFrame):
    return(
        data
        .assign(
                distance_type = pd.cut(data["distance"],bins=[0,5,10,15,25],
                                        right=False,labels=["short","medium","long","very_long"])
    ))


def perform_data_cleaning(data: pd.DataFrame):
    
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns,columns=columns_to_drop)
    )
    
    return cleaned_data.dropna()
    
    

if __name__ == "__main__":
    # data path for data
    DATA_PATH = "servicehub_dataset.csv"
    
    # read the data from path
    df = pd.read_csv(DATA_PATH)
    print('Data loaded successfuly')
    
    perform_data_cleaning(df)
