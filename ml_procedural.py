import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import os
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data():
    file_name = "data.csv"
    file_path = os.path.join(os.getcwd(), file_name)
    
    if not os.path.exists(file_path):
        print(f"File '{file_name}' not found in current directory, trying full path...")
        file_path = "D:\AURO\AURO workplace\VS Code\Python\Sem 3\DataSet Analysis\data.csv"
    
    try:
        data = pd.read_csv(file_path)
        print("File loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None

# Function to display data
def display_data(data):
    try:
        num_rows = int(input("How many rows would you like to print? (up to 200): "))
        if num_rows > 200:
            num_rows = 200
        print(data.head(num_rows))
    except ValueError:
        print("Please enter a valid number.")

# Preprocess the data (handle missing values and label encoding)
def preprocess_data(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # Drop 'prev_sold_date' if present
    if 'prev_sold_date' in data.columns:
        data.drop(columns=['prev_sold_date'], inplace=True)
    
    # Encode categorical columns
    for column in ['status', 'city', 'state']:
        if column in data.columns:
            data[column] = data[column].astype('category').cat.codes
    
    return data

# Function for Linear Regression
def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Linear Regression MSE: {mse}")
    
    # Plotting Actual vs Predicted Values
    plt.scatter(X_test['price'], y_test, color='blue', label="Actual")
    plt.scatter(X_test['price'], predictions, color='red', label="Predicted")
    
    # Plotting the Regression Line
    plt.plot(X_test['price'], predictions, color='red', label="Regression Line")
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Price')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

    return mse

# Function for KNN
def knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"KNN Accuracy: {accuracy}")
    
    # Plot for KNN
    plt.scatter(X_test['price'], y_test, color='blue', label='Actual')
    plt.scatter(X_test['price'], predictions, color='red', label='Predicted')
    plt.title('KNN: Actual vs Predicted')
    plt.legend(loc='upper left')
    plt.show()

# Function for K-Means Regression
def kmeans_regression(X, y, n_clusters=2):
    # Fit the KMeans model
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    
    # Get the cluster labels
    labels = model.labels_
    print(f'K-Means Clustering Labels: {np.unique(labels)}')
    
    # Plotting K-Means Clustering output
    for cluster in np.unique(labels):
        # Get indices of the samples belonging to this cluster
        cluster_indices = np.where(labels == cluster)[0]
        X_cluster = X.iloc[cluster_indices]  # Use .iloc to get the rows
        plt.scatter(X_cluster['price'], X_cluster['house_size'], label=f'Cluster {cluster}')

    plt.title('K-Means Clustering')
    plt.xlabel('Price')
    plt.ylabel('House Size')
    plt.legend(loc='upper left')  # Specifying legend location
    plt.show()

# Function to compare two algorithms
def compare_two_algorithms(X_train, X_test, y_train, y_test):
    print("1. Linear Regression\n2. KNN")
    algo1 = int(input("Choose the first algorithm to compare: "))
    algo2 = int(input("Choose the second algorithm to compare: "))

    # Run the first algorithm
    if algo1 == 1:
        linear_regression(X_train, X_test, y_train, y_test)
    elif algo1 == 2:
        knn(X_train, X_test, y_train, y_test)
    
    # Run the second algorithm
    if algo2 == 1:
        linear_regression(X_train, X_test, y_train, y_test)
    elif algo2 == 2:
        knn(X_train, X_test, y_train, y_test)

# Function to compare all algorithms
def compare_all_algorithms(X_train, X_test, y_train, y_test):
    print("Comparing all algorithms...")
    
    # Linear Regression
    linear_regression(X_train, X_test, y_train, y_test)

    # KNN
    knn(X_train, X_test, y_train, y_test)

    # K-Means Regression
    kmeans_regression(X_train, y_train)

# Main Menu function
def menu():
    data = load_data()
    if data is None:
        return

    data = preprocess_data(data)

    # Split data into features and target variable
    X = data[['price', 'bed', 'bath', 'acre_lot', 'house_size']]  # Update with actual feature columns
    y = data['status']  # Target variable for classification

    # Use 50-60% of the dataset for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    while True:
        print("\nMain Menu:")
        print("1. Display data")
        print("2. Choose algorithm")
        print("3. Compare two algorithms")
        print("4. Compare all algorithms")
        print("5. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            display_data(data)

        elif choice == 2:
            print("1. Linear Regression\n2. KNN\n3. K-Means Regression")
            algo_choice = int(input("Choose an algorithm: "))
            if algo_choice == 1:
                linear_regression(X_train, X_test, y_train, y_test)
            elif algo_choice == 2:
                knn(X_train, X_test, y_train, y_test)
            elif algo_choice == 3:
                kmeans_regression(X_train, y_train)
        
        elif choice == 3:
            compare_two_algorithms(X_train, X_test, y_train, y_test)
        
        elif choice == 4:
            compare_all_algorithms(X_train, X_test, y_train, y_test)
        
        elif choice == 5:
            print("Exiting program.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
