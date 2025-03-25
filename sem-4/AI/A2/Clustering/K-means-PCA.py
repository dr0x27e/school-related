# Imports:
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import plotly.express as px




# Loadning the dataset using pandas:
df = pd.read_csv("./data/apple_quality.csv", skipfooter=1, engine="python")

df = df.drop(columns=["A_id", "Quality"])

# Scaling the data points:

# Initializing a scaler:
scaler = StandardScaler()

# Scaling
df_scaled = scaler.fit_transform(df)






# Finding the optimal K value using elbow method:

# Within cluster sum of squares:
wcss = []
loop_range = range(1, 11)

# Test different values:
for k in loop_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method:
plt.figure(figsize=(8, 5))
plt.plot(loop_range, wcss, marker='o', linestyle='-')
plt.xlabel("Number of clusters: ")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()







# Trainig the model NOPCA:

# K
K = 3

# Training the model on the 7 dimensional data.
kmeans = KMeans(n_clusters=K)
kmeans.fit(df_scaled)

# Fetching labels and centroids:
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Creating an array with the headers of the dataframe.
columns = np.array(["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"])

# Creating a 7 by 3 plot:
fig, axes = plt.subplots(7, 3, figsize=(10, 10))

# Flattening axes for easy indexing:
axes = axes.flatten()

index = 0
    
# iterating through every possibility:
for n in range(7):
    for a in range(1 + n, 7):

        # Scatter plot for the current cluster:
        ax = axes[index]
        ax.scatter(df_scaled[:, n], df_scaled[:, a], c=labels, cmap='viridis', marker='o')

        # Plotting curretn centroids:
        ax.scatter(centroids[:, n], centroids[:, a], c='red', marker='x', s=100)

        # Setting labels:
        ax.set_xlabel(columns[n])
        ax.set_ylabel(columns[a])
        ax.set_title(f'Clusters for {columns[n]} and {columns[a]}')
        
        # Increasing the index.
        index += 1

plt.tight_layout()
plt.show()






# PCA:

# Number of components to keep 90% of variance:
n_comp = 6

# Applying PCA:
# And reduce df_scaled to the right size:
pca = PCA(n_components=n_comp)
df_pca = pca.fit_transform(df_scaled)





# Getting the explaned variance ration for each column
explained_variance_ratio = pca.explained_variance_ratio_

# Printing the vcariance
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Column {columns[i]}, captured: {ratio * 100} %")

# Plotting the variance:
plt.figure(figsize=(8, 5))
plt.bar(columns[:-1], explained_variance_ratio, color="blue")

plt.xlabel("Principal Components")
plt.ylabel("Variance")
plt.show()





# PCA-Transformed Data
plt.figure(figsize=(8, 5))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c="red")
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.title("New Dataset")
plt.show()





# Comparing the old dataset to the new one:

# Making the new and old dataset into a dataframe for better printing:
df_new = pd.DataFrame(df_pca, columns=columns[:-1])
df_old = pd.DataFrame(df_scaled, columns=columns)



# Printing the two datasets:
print("New Dataframe: ")
print(df_new)

print("\nOld DataFrane")
print(df_old)


# Training the KMeans on the new dataset to see the differance between old and new:
kmeans_pca = KMeans(n_clusters=K)
labels_pca = kmeans_pca.fit_predict(df_pca)

# Plotting the old and new clusters for weight and size:
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plotting old clusters for weight and size:
axes[0].scatter(df_scaled[:, 0], df_scaled[:, 1], c=labels, cmap="viridis", marker="o")
axes[0].set_xlabel(columns[0])
axes[0].set_ylabel(columns[1])
axes[0].set_title("Old Dataset Clusters")

# plotting new clusters for weight and size:
axes[1].scatter(df_pca[:, 0], df_pca[:, 1], c=labels_pca, cmap="viridis", marker="o")
axes[1].set_xlabel(columns[0])
axes[1].set_ylabel(columns[1])
axes[1].set_title("New Dataset Clusters")

plt.tight_layout()
plt.show()





# Logistic regression models on old and new dataset:

# Fetching the dataframe again as we drop "Quality" in an older task:
df = pd.read_csv("./data/apple_quality.csv", skipfooter=1, engine="python")

# Fetching Dependent variables:
Y = df["Quality"]

# Creating logistic regression function:
def logRes(dataSet):

    # Splitting the data into training and testing:
    X_train, X_test, Y_train, Y_test = train_test_split(dataSet, Y, test_size=0.2, random_state=42)
    
    # Creating a model:
    log_reg = LogisticRegression()
    
    # Training the model on the data:
    log_reg.fit(X_train, Y_train)

    # Return the predictions:
    return log_reg.predict(X_test), Y_test


# df_scaled and df_pca already only holds independent variables.

# Storing the predictions:
predict_old, y_test_old = logRes(df_scaled)
predict_new, y_test_new = logRes(df_pca)

# Calculating the accuract of the predictions using sklearn.metric.accuracy_score:
print(f"Accuracy of the Old dataset (Without PCA): {accuracy_score(y_test_old, predict_old)}")
print(f"Accuracy of the New dataset (With PCA)   : {accuracy_score(y_test_new, predict_new)}")






# Histograms and Box plots

# Dropping Non-numerical values and useless ones such as "A_id"
df = df.drop(columns=["A_id", "Quality"])

# Plotting Histogram:
plt.figure(figsize=(16, 8))
for i, col in enumerate(df):
    plt.subplot(1, 7, i + 1)
    plt.hist(df[col], bins=30, edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Plotting Box to detect outliers:
plt.figure(figsize=(16, 8))
for i, col in enumerate(df):
    plt.subplot(1, 7, i + 1)
    plt.boxplot(df[col], vert=True, patch_artist=True, boxprops=dict(facecolor="blue"))
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()






# Correlation heatmap

# Using pandas to find the correlations between the data:
correlation_matrix = df.corr()

# Plotting the heatmap:
# Using seaborn heatmap as its very quick and easy:
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm")
plt.title("Correlation Matrix Coefficients")
plt.show()







# Bar chart and pie chart

# resuing Y <- dependant variables from logistic regression task
count = Y.value_counts()

# Bar chart:
plt.figure(figsize=(8, 6))
plt.bar(count.index, count.values, color=["green", "red"])
plt.xlabel("Quality")
plt.ylabel("Number of apples")
plt.title("Number of good vs bad apples")
plt.show()

# Pie chart:
plt.figure(figsize=(8, 6))
plt.pie(count.values, labels=count.index, autopct="%1.1f%%", colors=["green", "red"])
plt.title("Number of good vs bad apples")
plt.show()






# Pair plot and interactive scatterplot.

# Pair Plot:
sns.pairplot(df)
plt.show()

df = pd.read_csv("./data/apple_quality.csv", skipfooter=1, engine="python")
df.drop(columns=["Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"])

# Interactive scatterplot:
fig = px.scatter(df, x="Size", y="Weight", color="Quality",
    title="interactive",
    labels={"Size": "Size", "Weight": "Weight", "Quality": "Quailty"}
)

fig.show()
