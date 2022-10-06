import pandas as pd
import numpy as np
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

def frequency_encode_stations(data,features):
    C,S = frequency_encoding_by_usertype("start station id",data)
    data["start customer freq"] = C
    data["start subscriber freq"]= S
    C,S = frequency_encoding_by_usertype("start station id",data)
    data["stop customer freq"] = C
    data["stop subscriber freq"]= S
    features = features + ["start customer freq","start subscriber freq","stop customer freq","stop subscriber freq"]
    return data,features


def load_data(data_path,features,preprocess=None,scaler=None,features_to_scale=None,fit_scaler=False,label="usertype"):
    """
    Load data from data_path, optionally preprocess it, and return only the columns in features.
    
    If provided, preprocess should be a callable that takes data and a list of features, processes the data (which can involve adding new features),
    and returns the processed data and the new list of features.
    """
    data = pd.read_parquet(data_path,engine="pyarrow")
    if preprocess:
        data,features = preprocess(data,features)
    unused = [c for c in data.columns if c not in features]
    #Conversion to float32 because otherwhise the decision tree trainer will copy the data
    X = data.drop(columns=unused).astype(np.float32)
    Y = data["usertype"].copy()
    if scaler:
        if fit_scaler:
            scaler.fit(X[features_to_scale])
        X[features_to_scale] = scaler.transform(X[features_to_scale])  
    Y=(Y=="Customer").astype(np.float32)
    return X,Y

def train(data_path,clf,features,preprocess=None,scaler=None,features_to_scale=None,fit_scaler=False):
    """
    Load data from data_path, optionally preprocess it, and train the given classifier on it.
    """
    X_train,Y_train = load_data(data_path,features,preprocess=preprocess,scaler=scaler,features_to_scale=features_to_scale,fit_scaler=fit_scaler)
    clf = clf.fit(X_train,Y_train)
    acc,conf,mcc = evaluate_model(clf,X_train,Y_train)
    return clf,X_train.columns

def evaluate_model(model,X,Y,verbose=True):
    """Print some summary statistics about the performance of model on the given data."""
    #TODO: Add uncertainty estimates about these
    Y_pred = model.predict(X)
    acc = accuracy_score(Y,Y_pred)
    confusion= confusion_matrix(Y,Y_pred,normalize="true")
    MCC = matthews_corrcoef(Y,Y_pred)
    if verbose:
        print(f"Accuracy: {acc}")
        print(f"Confusion: ")
        print(confusion)
        print(f"MCC: {MCC}")
    return acc,confusion,MCC

def validate(model,data_path,features,preprocess=None,scaler=None,features_to_scale=None,fit_scaler=False):
    """Load data from path and evaluate_model on it. """
    X_val,Y_val = load_data(data_path,features,preprocess=preprocess,scaler=scaler,features_to_scale=features_to_scale,fit_scaler=fit_scaler)
    acc,conf,MCC=evaluate_model(model,X_val,Y_val)
    return acc,conf,MCC