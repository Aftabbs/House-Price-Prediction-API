import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    df['Age'] = 2024 - df['Year Built']  
    df['Date Sold'] = pd.to_datetime(df['Date Sold'])
    df['Year Sold'] = df['Date Sold'].dt.year
    df['Month Sold'] = df['Date Sold'].dt.month
    df.drop(columns=['Property ID', 'Year Built', 'Date Sold'], inplace=True)  
    
    num_features = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Year Sold', 'Month Sold']
    cat_features = ['Location', 'Condition', 'Type']
    
    for col in num_features:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in cat_features:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def prepare_data(df):
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    num_features = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Year Sold', 'Month Sold']
    cat_features = ['Location', 'Condition', 'Type']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    return X, y, preprocessor

def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    print("Model Performance:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))
    
    joblib.dump(model_pipeline, "house_price_model.pkl")  
    print("Model Saved!")

file_path = "Case Study 1 Data.xlsx" 
if not os.path.exists("house_price_model.pkl"):
    print("Model not found. Training a new model...")
    df = load_data(file_path)
    df = preprocess_data(df)
    X, y, preprocessor = prepare_data(df)
    train_model(X, y, preprocessor)
else:
    print("Model found. Loading existing model...")

app = FastAPI()
model = joblib.load("house_price_model.pkl")

class HouseFeatures(BaseModel):
    Location: str
    Size: float
    Bedrooms: int
    Bathrooms: int
    Condition: str
    Type: str
    Year_Sold: int
    Month_Sold: int
    Age: int

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API! Use /predict to get price predictions."}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.model_dump()])

    # Pydantic deprecation warning.
    # data = pd.DataFrame([features.dict()])
    data = data.rename(columns={"Year_Sold": "Year Sold", "Month_Sold": "Month Sold"})

    prediction = model.predict(data)[0]
    return {"Predicted Price": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
