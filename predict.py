import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def load_data(data_path):
    try:
        columns = ['timestamp', 'OZONE', 'NO2', 'temp', 'humidity', 'no2op1', 'no2op2', 'o3op1', 'o3op2']
        data = pd.read_csv(data_path, names=columns, header=0)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def preprocess_data(data):
    try:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
                
        data.drop(columns=['timestamp' ,'hour'], inplace=True)
        data['temp * humidity'] = data['temp'] * data['humidity']
        
        X = data[['no2op1', 'no2op2', 'o3op1', 'o3op2','temp', 'humidity', 'cos_hour', 'temp * humidity']]
        Y_O3_desired = data['OZONE']
        Y_NO2_desired = data['NO2']
        return X, Y_O3_desired, Y_NO2_desired
        
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None
    
def predict(model, X):
    try:
        Y_NO2_predict = model['NO2'].predict(X)
        Y_O3_predict = model['O3'].predict(X)
        Y_NO2_predict = pd.DataFrame(Y_NO2_predict, columns=['NO2'])
        Y_O3_predict = pd.DataFrame(Y_O3_predict, columns=['O3'])
        return Y_NO2_predict, Y_O3_predict
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None
    
def evaluate(predictions, true_values):
    try:
        predictions = np.array(predictions).flatten()
        true_values = np.array(true_values).flatten()
        mae = mean_absolute_error(true_values, predictions)
        return mae
    except Exception as e:
        print(f"Error evaluating predictions: {e}")
        return None
    
def main():
    print("Waiting for testing on data")
     
    model_path = "models/advanced_model.pkl"
    data_path = "data/dummy_test.csv"
   
    model = load_model(model_path)
    if model is None:
        return
    
    data = load_data(data_path)
    if data is None:
        return
    
    X, Y_O3_desired, Y_NO2_desired = preprocess_data(data)
    if X is None:
        return
    
    Y_NO2_predict, Y_O3_predict = predict(model, X)
    
    if Y_NO2_predict is None or Y_O3_predict is None:
        return
    print("Mean Absolute Error for NO2 predictions:")
    print(evaluate(Y_NO2_predict, Y_NO2_desired))
    print("Mean Absolute Error for O3 predictions:")
    print(evaluate(Y_O3_predict, Y_O3_desired))

        
if __name__ == "__main__":
    main()