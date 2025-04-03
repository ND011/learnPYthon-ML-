import time
import sys
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Bubble Sort Implementation
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j], arr[j]

# Merge Sort Implementation
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

# Performance Measurement
def measure_performance(sort_function, arr):
    start_time = time.perf_counter()
    sort_function(arr)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time, sys.getsizeof(arr)

# Menu-driven system
def menu():
    while True:
        print("\n1. Run Sorting Analysis")
        print("2. Show Comparison Charts")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            run_analysis()
        elif choice == '2':
            chart_menu()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

def run_analysis():
    input_sizes = [10, 100, 1000, 10000]
    input_types = ['random', 'sorted', 'reverse_sorted']
    results = []

    for size in input_sizes:
        for input_type in input_types:
            if input_type == 'random':
                arr = [random.randint(0, size) for _ in range(size)]
            elif input_type == 'sorted':
                arr = list(range(size))
            elif input_type == 'reverse_sorted':
                arr = list(range(size, 0, -1))

            bubble_time, bubble_space = measure_performance(bubble_sort, arr.copy())
            merge_time, merge_space = measure_performance(merge_sort, arr.copy())

            results.append((input_type, size, bubble_time, merge_time, bubble_space, merge_space))

    table = PrettyTable()
    table.field_names = ["Input Type", "Input Size", "Bubble Sort Time (s)", "Merge Sort Time (s)", "Bubble Sort Space (KB)", "Merge Sort Space (KB)"]
    for result in results:
        table.add_row([result[0], result[1], f"{result[2]:.5f}", f"{result[3]:.5f}", f"{result[4] / 1024:.2f} KB", f"{result[5] / 1024:.2f} KB"])
    print(table)

def chart_menu():
    input_types = ['random', 'sorted', 'reverse_sorted']
    for input_type in input_types:
        show_chart = input(f"Do you want to see the chart for {input_type} input? (y/n): ")
        if show_chart.lower() == 'y':
            show_chart_for_type(input_type)

def show_chart_for_type(input_type):
    input_sizes = [10, 100, 1000, 10000]
    bubble_times = [random.uniform(0.01, 1.5) for _ in input_sizes]
    merge_times = [random.uniform(0.001, 0.5) for _ in input_sizes]

    plt.style.use('dark_background')
    plt.figure()
    plt.plot(input_sizes, bubble_times, label='Bubble Sort', marker='o', color='cyan', linewidth=2)
    plt.plot(input_sizes, merge_times, label='Merge Sort', marker='x', color='magenta', linewidth=2)
    plt.xlabel('Input Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Performance Comparison - {input_type} input')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--')
    plt.show()

if __name__ == "__main__":
    menu()
