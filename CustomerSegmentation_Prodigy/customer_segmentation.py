import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load data
customers = pd.read_csv("Mall_Customers.csv")

# Step 2: Select relevant numeric columns
features = customers[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 3: Find best number of clusters (Elbow Method)
distortions = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, init="k-means++", random_state=1)
    model.fit(features)
    distortions.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), distortions, marker='o', linewidth=2)
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion (WCSS)")
plt.grid(True)
plt.show()

# Step 4: Build final model (choose optimal clusters, e.g., 5)
final_model = KMeans(n_clusters=5, init="k-means++", random_state=1)
customers["Group"] = final_model.fit_predict(features)

# Step 5: Inspect results
centroids = pd.DataFrame(final_model.cluster_centers_, columns=features.columns)
print("\nCluster Centers:\n")
print(centroids.round(2))

# Step 6: Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(features["Annual Income (k$)"], features["Spending Score (1-100)"],
            c=customers["Group"], cmap="plasma", s=50)
plt.scatter(centroids["Annual Income (k$)"], centroids["Spending Score (1-100)"],
            color="black", marker="X", s=200, label="Centroids")
plt.title("Customer Segmentation with K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Step 7: Save output
customers.to_csv("segmented_customers.csv", index=False)
print("\nSegmented customer data saved to segmented_customers.csv")
