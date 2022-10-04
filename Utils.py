import pandas as pd

def load_data(path):
    dtype = {"tripduration":"Int64", #Note that this accepts nan values
              "start station id":"Int64",
              "start station name":"category",
              "start station latitude":"Float64",
              "start station longitude":"Float64",
              "end station id":"Int64",
              "end station name":"category",
              "end station latitude":"Float64",
              "end station longitude":"Float64",
              "bikeid":"Int64",
              "usertype":"category",
              "birth year":"Int64",
              "gender":"category"}
    data = pd.read_csv(path,parse_dates=["starttime","stoptime"])
    data = data.astype(dtype,copy=False) #Because reading category dtypes does not work out of the box
    return data

def print_memory_usage(data):
    memory = data.memory_usage(index=True,deep=True).sum()
    print(f"The dataframe needs {memory/1e9:.3} GB of memory") 