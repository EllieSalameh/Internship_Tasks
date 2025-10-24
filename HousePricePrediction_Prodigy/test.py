import pandas as pd
import joblib

# Step 1: Load your saved model
model = joblib.load("house_price_model.pkl")

# Step 2: Load the test dataset
test_data = pd.read_csv("test.csv")

# Step 3: Check which columns exist
print("Columns in test.csv:")
print(test_data.columns.tolist())

# Step 4: Use only the features your model trained on
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Step 5: Predict the prices
predicted_prices = model.predict(X_test)

# Step 6: Combine with IDs to see results
results = pd.DataFrame({
    'Id': test_data['Id'],
    'Predicted_SalePrice': predicted_prices
})

# Step 7: Save predictions to a file
results.to_csv("predicted_house_prices.csv", index=False)

# Step 8: Show a preview
print("\nPredictions completed! Here are the first few rows:")
print(results.head())
