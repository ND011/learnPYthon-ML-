# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import os

# Step 1: Check the current working directory
print("Current Working Directory:", os.getcwd())

# Step 2: Define the file path
file_name = "data.csv"
file_path = os.path.join(os.getcwd(), file_name)

# If the file isn't in the current directory, provide the full path.
if not os.path.exists(file_path):
    print(f"File '{file_name}' not found in current directory, trying full path...")
    file_path = "d:/cd/python/data.csv"

# Step 3: Read the CSV file
try:
    data = pd.read_csv(file_path)
    print("File loaded successfully.")

    # Step 4: Handle missing data
    # Fill missing numeric values with the median of each column
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # Drop 'prev_sold_date' if not needed
    if 'prev_sold_date' in data.columns:
        data.drop(columns=['prev_sold_date'], inplace=True)

    # Step 5: Encode categorical data (status, city, state) using Label Encoding
    for column in ['status', 'city', 'state']:
        data[column] = data[column].astype('category').cat.codes

    # Step 6: Feature selection (selecting only the relevant columns)
    X = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]  # Select numeric features
    y = data['status']  # Assuming 'status' is the target variable for classification tasks

    # Step 7: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def display_data():
        rows_to_display = int(input("How many rows of data would you like to print? "))
        print(data.head(rows_to_display))

    def linear_regression():
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        y_pred_lin = linear_reg.predict(X_test)
        mse_lin = mean_squared_error(y_test, y_pred_lin)
        print(f'Linear Regression MSE: {mse_lin}')

        plt.figure(figsize=(10, 5))
        plt.bar(['MSE'], [mse_lin], color='blue')
        plt.title('Linear Regression Performance')
        plt.ylabel('Mean Squared Error')
        plt.show(block=False)  # Use block=False to prevent blocking the terminal

    def knn_classifier():
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        print(f'KNN Accuracy: {acc_knn}')

        plt.figure(figsize=(10, 5))
        plt.bar(['Accuracy'], [acc_knn], color='green')
        plt.title('KNN Performance')
        plt.ylabel('Accuracy')
        plt.show(block=False)  # Use block=False to prevent blocking the terminal

    def kmeans_clustering():
        kmeans = KMeans(n_clusters=2)  # Assume 2 clusters for binary classification
        kmeans.fit(X)
        kmeans_labels = kmeans.labels_
        print(f'K-Means Clustering Labels: {np.unique(kmeans_labels)}')

        plt.figure(figsize=(10, 5))
        plt.hist(kmeans_labels, bins=2, color='orange', alpha=0.7)
        plt.title('K-Means Clustering Distribution')
        plt.xlabel('Clusters')
        plt.ylabel('Count')
        plt.show(block=False)  # Use block=False to prevent blocking the terminal

    def compare_all_algorithms():
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        y_pred_lin = linear_reg.predict(X_test)
        mse_lin = mean_squared_error(y_test, y_pred_lin)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)

        # Plotting comparison
        plt.figure(figsize=(10, 5))
        plt.bar(['Linear Regression MSE', 'KNN Accuracy'], [mse_lin, acc_knn], color=['blue', 'green'])
        plt.title('Algorithm Performance Comparison')
        plt.ylabel('Performance Metric')
        plt.show(block=False)  # Use block=False to prevent blocking the terminal

    # Menu driven interface
    def menu():
        while True:
            print("\nMain Menu:")
            print("1. Display data")
            print("2. Choose algorithm")
            print("3. Compare two algorithms")
            print("4. Compare all algorithms")
            print("5. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                display_data()
            elif choice == '2':
                print("\nChoose an algorithm:")
                print("1. Linear Regression")
                print("2. KNN")
                print("3. K-Means Clustering")
                algo_choice = input("Choose an algorithm: ")

                if algo_choice == '1':
                    linear_regression()
                elif algo_choice == '2':
                    knn_classifier()
                elif algo_choice == '3':
                    kmeans_clustering()
                else:
                    print("Invalid choice. Please select again.")
            elif choice == '3':
                print("Comparing two algorithms is not implemented yet.")
            elif choice == '4':
                compare_all_algorithms()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select again.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
