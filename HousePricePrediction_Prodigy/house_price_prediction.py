
# House Price Prediction
# Using Linear Regression (with data cleaning)

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('train.csv')
print("Dataset loaded successfully")


#columns we needed
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

#Check and clean missing data
print("\nChecking for missing values...")
print(data.isnull().sum()) 

# Drop any rows that have missing values
data = data.dropna()
print("Missing values removed, Remaining rows:", len(data))

#Separate features and target 
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

#Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n Model training completed")

#Save the trained model
joblib.dump(model, 'house_price_model.pkl')


#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Results:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 3))

#Visualize actual vs predicted prices
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

#Load the saved model and use it again
loaded_model = joblib.load('house_price_model.pkl')
print("\n Model loaded successfully")


#user input custom house details
print("\n Let's predict a custom house price!")
area = float(input("Enter the living area (in square feet): "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of full bathrooms: "))

# Make prediction
user_input = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = loaded_model.predict(user_input)
print(f"\n Estimated House Price: ${round(predicted_price[0], 2)}")