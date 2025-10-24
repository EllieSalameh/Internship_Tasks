import tkinter as tk
from tkinter import messagebox
import joblib

# Load the trained model
try:
    model = joblib.load('house_price_model.pkl')
except:
    messagebox.showerror("Error", "Model file not found! Please run the training script first.")
    exit()

# Function to predict the price
def predict_price():
    try:
        area = float(entry_area.get())
        bedrooms = int(entry_bedrooms.get())
        bathrooms = int(entry_bathrooms.get())

        # Predict using the model
        prediction = model.predict([[area, bedrooms, bathrooms]])
        price = round(prediction[0], 2)

        result_label.config(
            text=f"Estimated Price: ${price:,.2f}",
            fg="blue"
        )

    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers.")

# Create the main window
window = tk.Tk()
window.title("House Price Predictor")
window.geometry("360x320")
window.config(bg="#f8f9fa")

# Title label
tk.Label(
    window, text="House Price Prediction",
    font=("Helvetica", 16, "bold"),
    bg="#f8f9fa", fg="#003566"
).pack(pady=15)

# Input area
frame = tk.Frame(window, bg="#f8f9fa")
frame.pack(pady=10)

tk.Label(frame, text="Living Area (sq ft):", bg="#f8f9fa").grid(row=0, column=0, sticky="w", pady=5)
entry_area = tk.Entry(frame, width=20)
entry_area.grid(row=0, column=1, pady=5)

tk.Label(frame, text="Bedrooms:", bg="#f8f9fa").grid(row=1, column=0, sticky="w", pady=5)
entry_bedrooms = tk.Entry(frame, width=20)
entry_bedrooms.grid(row=1, column=1, pady=5)

tk.Label(frame, text="Bathrooms:", bg="#f8f9fa").grid(row=2, column=0, sticky="w", pady=5)
entry_bathrooms = tk.Entry(frame, width=20)
entry_bathrooms.grid(row=2, column=1, pady=5)

# Predict button
tk.Button(
    window, text="Predict Price", command=predict_price,
    bg="#0077b6", fg="white", font=("Arial", 12, "bold"),
    relief="raised", padx=8, pady=5
).pack(pady=15)

# Result label
result_label = tk.Label(window, text="", bg="#f8f9fa", font=("Arial", 12))
result_label.pack(pady=10)

# Run the window
window.mainloop()
