# House Price Prediction API

##  Overview
This project is an end-to-end **House Price Prediction** system built using **FastAPI** and **Random Forest Regressor**. It provides an API endpoint where users can input house features and receive a predicted price.

##  Features
- **Data Preprocessing & Feature Engineering**: Handles missing values, encodes categorical variables, and standardizes numerical features.
- **Machine Learning Model**: Uses a **Random Forest Regressor** for price prediction.
- **API Deployment**: Built using **FastAPI** and served with **Uvicorn**.
- **Automated Model Training**: If no trained model is found, the script automatically trains a new one.

##  Project Structure
```
├── Case Study 1 Data.xlsx  # dataset
├── house_price_model.pkl  # Trained model (saved after training)
├── Case Study Solution 1.py  # Main script containing API and model logic
├── readme.md  # Documentation
```

## 🛠 Installation & Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**
   ```bash
   python Case Study Solution 1.py
   ```
   The API will be available at: **http://127.0.0.1:8000**

## API Usage
### Predict House Price
**Endpoint:** `POST /predict`

**Request Format:**
```json
{
    "Location": "New York",
    "Size": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Condition": "Excellent",
    "Type": "Apartment",
    "Year_Sold": 2023,
    "Month_Sold": 5,
    "Age": 20
}
```

**Response:**
```json
{
    "Predicted Price": 450000.0
}
```

### **Exploratory Data Analysis (EDA) & Model Approach**  

1. **Feature Engineering:**  
   - Created **Age** feature by subtracting **Year Built** from 2024.  
   - Extracted **Year Sold** and **Month Sold** from **Date Sold**.  
   - Dropped unnecessary columns like **Property ID, Year Built, and Date Sold**.  

2. **Handling Missing Values:**  
   - Filled missing numerical values with the **median** of each column.  
   - Filled missing categorical values with the **mode** of each column.  

3. **Data Preprocessing using ColumnTransformer:**  

   - **StandardScaler**: Normalizes numerical features.  
   - **OneHotEncoder**: Converts categorical variables into numerical format.  
  
4. **Model Training using a Pipeline:**  

   - **Preprocessing & Model Training** combined into a single pipeline.  
   - **RandomForestRegressor** chosen for its ability to handle feature interactions and importance.  

5. **Model Evaluation Metrics:**  

   - **Mean Absolute Error (MAE)**  
   - **Root Mean Squared Error (RMSE)**  
   - **R² Score**  

6. **Model Deployment:**  
   - Saved the trained model using **Joblib**.  
   - Built a **FastAPI** service for real-time price predictions.  


## 📈 Model Performance
| Metric       | Value  |
|-------------|--------|
| MAE         | 15,320 |
| RMSE        | 22,540 |
| R² Score    | 0.87   |


## 🏆 Acknowledgments
- **FastAPI** for API development
- **Scikit-learn** for machine learning
- **Pandas & NumPy** for data manipulation



