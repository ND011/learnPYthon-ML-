import time
import sys
import pandas as pd
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def measure_performance(sort_function, arr):
    start_time = time.perf_counter()
    sort_function(arr)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time, sys.getsizeof(arr)

def run_sorting_analysis(dataset_types):
    for dataset_type in dataset_types:
        data = df[dataset_type].dropna().astype(int).tolist()
        print(f"\nDataset Type: {dataset_type}")
        print("Original Data:", data[:10], "...")

        bubble_sorted = data.copy()
        bubble_time, bubble_space = measure_performance(bubble_sort, bubble_sorted)
        print("Bubble Sort Sorted Data:", bubble_sorted[:10], "...")
        print(f"Bubble Sort Time: {bubble_time:.5f} sec, Space: {bubble_space} bytes")

        merge_sorted = data.copy()
        merge_time, merge_space = measure_performance(merge_sort, merge_sorted)
        print("Merge Sort Sorted Data:", merge_sorted[:10], "...")
        print(f"Merge Sort Time: {merge_time:.5f} sec, Space: {merge_space} bytes")

def plot_comparison_chart(dataset_types):
    sizes = [len(df[dataset_type].dropna()) for dataset_type in dataset_types]
    bubble_times = []
    merge_times = []

    for dataset_type in dataset_types:
        data = df[dataset_type].dropna().astype(int).tolist()
        bubble_sorted = data.copy()
        bubble_time, _ = measure_performance(bubble_sort, bubble_sorted)
        bubble_times.append(bubble_time)

        merge_sorted = data.copy()
        merge_time, _ = measure_performance(merge_sort, merge_sorted)
        merge_times.append(merge_time)

    plt.plot(sizes, bubble_times, label='Bubble Sort', marker='o')
    plt.plot(sizes, merge_times, label='Merge Sort', marker='x')
    plt.xlabel('Dataset Size')
    plt.ylabel('Time (seconds)')
    plt.title('Sorting Performance Comparison')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('shree.csv')
    
    while True:
        print("\n1. Small Data\n2. Medium Data\n3. Large Data\n4. Exit")
        choice = input("Select dataset size: ")

        if choice == '1':
            dataset_types = ['small_random', 'small_sorted', 'small_reverse_sorted', 'small_duplicates', 'small_partial_sorted']
        elif choice == '2':
            dataset_types = ['medium_random', 'medium_sorted', 'medium_reverse_sorted', 'medium_duplicates', 'medium_partial_sorted']
        elif choice == '3':
            dataset_types = ['large_random', 'large_sorted', 'large_reverse_sorted', 'large_duplicates', 'large_partial_sorted']
        elif choice == '4':
            break
        else:
            print("Invalid choice. Try again.")
            continue

        run_sorting_analysis(dataset_types)
        see_chart = input("Do you want to see the chart? (y/n): ")
        if see_chart.lower() == 'y':
            plot_comparison_chart(dataset_types)
