# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import os

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("File loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None

# Function to display data
def display_data(data):
    num_rows = int(input("How many rows of data do you wish to print? "))
    print(data.head(num_rows))

# Function for Linear Regression
def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Linear Regression MSE: {mse}")
    
    # Plotting Actual vs Predicted Values
    plt.scatter(y_test, predictions, color='blue', label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal Fit")
    
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()  # Show the plot

# Function for K-Nearest Neighbors
def knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'KNN Accuracy: {accuracy}')
    
    # Plotting Accuracy
    plt.bar(['KNN'], [accuracy])
    plt.ylim([0, 1])
    plt.title('KNN Accuracy')
    plt.ylabel('Accuracy')
    plt.show()  # Show the plot

# Function for K-Means Clustering
def kmeans_clustering(X):
    model = KMeans(n_clusters=2)  # Assume 2 clusters for binary classification
    model.fit(X)
    labels = model.labels_
    print(f'K-Means Clustering Labels: {np.unique(labels)}')

    # Plotting Clusters
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
    plt.title('K-Means Clustering')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.show()  # Show the plot

# Function to compare two algorithms
def compare_two_algorithms(X_train, X_test, y_train, y_test, algo1, algo2):
    # Initialize performance metrics
    results = {}

    # Compare Linear Regression
    if algo1 == 'Linear Regression':
        model1 = LinearRegression()
        model1.fit(X_train, y_train)
        predictions1 = model1.predict(X_test)
        mse1 = mean_squared_error(y_test, predictions1)
        results['Linear Regression'] = mse1
        print(f"Linear Regression MSE: {mse1}")

    elif algo1 == 'KNN':
        model1 = KNeighborsClassifier(n_neighbors=5)
        model1.fit(X_train, y_train)
        predictions1 = model1.predict(X_test)
        accuracy1 = accuracy_score(y_test, predictions1)
        results['KNN'] = accuracy1
        print(f"KNN Accuracy: {accuracy1}")

    # Compare K-Means Clustering
    if algo2 == 'Linear Regression':
        model2 = LinearRegression()
        model2.fit(X_train, y_train)
        predictions2 = model2.predict(X_test)
        mse2 = mean_squared_error(y_test, predictions2)
        results['Linear Regression'] = mse2
        print(f"Linear Regression MSE: {mse2}")

    elif algo2 == 'KNN':
        model2 = KNeighborsClassifier(n_neighbors=5)
        model2.fit(X_train, y_train)
        predictions2 = model2.predict(X_test)
        accuracy2 = accuracy_score(y_test, predictions2)
        results['KNN'] = accuracy2
        print(f"KNN Accuracy: {accuracy2}")

    # Plotting results
    plt.bar(results.keys(), results.values())
    plt.title('Comparison of Algorithms')
    plt.ylabel('Performance Metric')
    plt.show()  # Show the plot

# Function to compare all algorithms
def compare_all_algorithms(X_train, X_test, y_train, y_test):
    # Initialize performance metrics
    results = {}

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results['Linear Regression'] = mean_squared_error(y_test, y_pred)

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    results['KNN'] = accuracy_score(y_test, knn_pred)

    # K-Means Clustering (for comparison, using the same X)
    kmeans_model = KMeans(n_clusters=2)
    kmeans_model.fit(X_train)  # Using training data for clustering
    kmeans_labels = kmeans_model.labels_
    results['K-Means Clustering'] = 'Labels generated, check console'

    # Print results
    for algo, result in results.items():
        print(f'{algo}: {result}')

    # Identify the best fit algorithm
    best_fit_algo = min(
        (k for k in results.keys() if isinstance(results[k], float)),
        key=results.get
    )
    best_fit_value = results[best_fit_algo]
    print(f"\nBest Fit Algorithm: {best_fit_algo} with a score of {best_fit_value}")

    # Plot results
    plt.bar(results.keys(), [result if isinstance(result, float) else 0 for result in results.values()])
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Performance Metric')
    plt.show()  # Show the plot

# Main function to run the program
def menu():
    # Define file path
    file_name = "data.csv"
    file_path = os.path.join(os.getcwd(), file_name)

    # Load data
    data = load_data(file_path)
    if data is None:
        return

    # Handle missing data
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # Encode categorical data (status, city, state) using Label Encoding
    for column in ['status', 'city', 'state']:
        data[column] = data[column].astype('category').cat.codes

    # Feature selection (selecting only the relevant columns)
    X = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]  # Select numeric features
    y = data['status']  # Assuming 'status' is the target variable for classification tasks

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    while True:
        print("\nMain Menu:")
        print("1. Display data")
        print("2. Choose algorithm")
        print("3. Compare two algorithms")
        print("4. Compare all algorithms")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            display_data(data)
        elif choice == '2':
            print("Choose an algorithm:")
            print("1. Linear Regression")
            print("2. KNN")
            print("3. K-Means Clustering")
            algo_choice = input("Choose an algorithm: ")

            if algo_choice == '1':
                linear_regression(X_train, X_test, y_train, y_test)
            elif algo_choice == '2':
                knn(X_train, X_test, y_train, y_test)
            elif algo_choice == '3':
                kmeans_clustering(X)
            else:
                print("Invalid choice.")
        elif choice == '3':
            print("Choose first algorithm:")
            print("1. Linear Regression")
            print("2. KNN")
            algo1_choice = input("Choose first algorithm: ")

            print("Choose second algorithm:")
            print("1. Linear Regression")
            print("2. KNN")
            algo2_choice = input("Choose second algorithm: ")

            algo1 = 'Linear Regression' if algo1_choice == '1' else 'KNN'
            algo2 = 'Linear Regression' if algo2_choice == '1' else 'KNN'
            compare_two_algorithms(X_train, X_test, y_train, y_test, algo1, algo2)
        elif choice == '4':
            compare_all_algorithms(X_train, X_test, y_train, y_test)
        elif choice == '5':
            break
        else:
            print("Invalid choice.")

# Run the menu
if __name__ == "__main__":
    menu()
