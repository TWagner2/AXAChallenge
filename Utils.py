import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

def print_memory_usage(data):
    memory = data.memory_usage(index=True,deep=True).sum()
    print(f"The dataframe needs {memory/1e9:.3} GB of memory") 

def frequency_encoding_by_usertype(column,data): #Encode by the frequency of customers and subscribers
    counts = data.groupby("usertype")[column].value_counts()
    counts = counts / counts.groupby("usertype").sum()
    Customer_count = data[column].map(counts["Customer"]).fillna(0).astype(float)
    Subscriber_count = data[column].map(counts["Subscriber"]).fillna(0).astype(float)
    return Customer_count, Subscriber_count

def frequency_encode_stations(X,data):
    C,S = frequency_encoding_by_usertype("start station id",data)
    X["start customer freq"] = C
    X["start subscriber freq"]= S
    C,S = frequency_encoding_by_usertype("start station id",data)
    X["stop customer freq"] = C
    X["stop subscriber freq"]= S
    return X

def evaluate_model(model,X,Y,verbose=True):
    #print some summary statistics about the model
    #TODO: Add uncertainty estimates about these
    Y_pred = model.predict(X)
    acc = accuracy_score(Y,Y_pred)
    confusion= confusion_matrix(Y,Y_pred,normalize="true")
    matthews = matthews_corrcoef(Y,Y_pred)
    if verbose:
        print(f"Accuracy: {acc}")
        print(f"Confusion: ")
        print(confusion)
        print(f"MCC: {matthews}")
    return acc,confusion
