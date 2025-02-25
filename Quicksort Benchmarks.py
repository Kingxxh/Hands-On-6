import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import sys

# Increase recursion limit for deep recursion cases
sys.setrecursionlimit(10000)


# Non-random pivot quicksort (always choosing first element as pivot)
def quicksort_nonrandom(arr: List[int], low: int = 0, high: int = None) -> List[int]:
    """
    Implementation of quicksort using the first element as pivot.

    Args:
        arr: List to be sorted
        low: Starting index
        high: Ending index

    Returns:
        Sorted list
    """
    # Create a copy of the array to avoid modifying the original
    if high is None:
        high = len(arr) - 1
        arr = arr.copy()

    if low < high:
        # Partition the array and get the pivot index
        pivot_index = partition_nonrandom(arr, low, high)

        # Recursively sort the sub-arrays
        quicksort_nonrandom(arr, low, pivot_index - 1)
        quicksort_nonrandom(arr, pivot_index + 1, high)

    return arr


def partition_nonrandom(arr: List[int], low: int, high: int) -> int:
    """
    Partition function for non-random pivot quicksort.
    Always chooses the first element as pivot.

    Args:
        arr: List to be partitioned
        low: Starting index
        high: Ending index

    Returns:
        Index of the pivot after partitioning
    """
    # Choose the first element as pivot
    pivot = arr[low]

    # Index of smaller element
    i = low

    for j in range(low + 1, high + 1):
        # If current element is smaller than the pivot
        if arr[j] < pivot:
            # Increment i
            i += 1
            # Swap arr[i] and arr[j]
            arr[i], arr[j] = arr[j], arr[i]

    # Swap the pivot element with the element at i
    arr[low], arr[i] = arr[i], arr[low]
    return i


# Random pivot quicksort
def quicksort_random(arr: List[int], low: int = 0, high: int = None) -> List[int]:
    """
    Implementation of quicksort using a random element as pivot.

    Args:
        arr: List to be sorted
        low: Starting index
        high: Ending index

    Returns:
        Sorted list
    """
    # Create a copy of the array to avoid modifying the original
    if high is None:
        high = len(arr) - 1
        arr = arr.copy()

    if low < high:
        # Partition the array and get the pivot index
        pivot_index = partition_random(arr, low, high)

        # Recursively sort the sub-arrays
        quicksort_random(arr, low, pivot_index - 1)
        quicksort_random(arr, pivot_index + 1, high)

    return arr


def partition_random(arr: List[int], low: int, high: int) -> int:
    """
    Partition function for random pivot quicksort.
    Chooses a random element as pivot.

    Args:
        arr: List to be partitioned
        low: Starting index
        high: Ending index

    Returns:
        Index of the pivot after partitioning
    """
    # Choose a random element as pivot and swap with the first element
    pivot_idx = random.randint(low, high)
    arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]

    # Now use the same partitioning logic as non-random version
    return partition_nonrandom(arr, low, high)


# Generate different test cases
def generate_best_case(n: int) -> List[int]:
    """
    Generate best case input for non-random quicksort.
    For first-element pivot quicksort, the best case is when the pivot
    divides the array into two nearly equal parts after partitioning.
    A median-of-medians approach would be ideal, but for simplicity,
    we'll create an already-sorted array and then rearrange it slightly.

    Args:
        n: Size of the array

    Returns:
        List of integers representing best case
    """
    if n <= 1:
        return list(range(n))

    # Start with a sorted array
    arr = list(range(n))

    # Rearrange elements to create a favorable partitioning pattern
    # One approach is to place the median at the first position
    median_idx = n // 2
    arr[0], arr[median_idx] = arr[median_idx], arr[0]

    return arr


def generate_worst_case(n: int) -> List[int]:
    """
    Generate worst case input for non-random quicksort.
    For first-element pivot, the worst case is when the array is already
    sorted in reverse order, as it leads to the most unbalanced partitioning.

    Args:
        n: Size of the array

    Returns:
        List of integers representing worst case
    """
    return list(range(n, 0, -1))  # Descending order


def generate_average_case(n: int) -> List[int]:
    """
    Generate average case input for quicksort by randomly shuffling elements.

    Args:
        n: Size of the array

    Returns:
        List of integers representing average case
    """
    arr = list(range(n))
    random.shuffle(arr)
    return arr


# Benchmark function with safeguards against excessive recursion
def benchmark(sort_func: Callable, input_generator: Callable, sizes: List[int],
              repeats: int = 3) -> List[Tuple[int, float]]:
    """
    Benchmark a sorting function with various input sizes.

    Args:
        sort_func: Sorting function to benchmark
        input_generator: Function to generate test cases
        sizes: List of input sizes to test
        repeats: Number of times to repeat each test for averaging

    Returns:
        List of (size, time) tuples
    """
    results = []

    for n in sizes:
        total_time = 0

        for _ in range(repeats):
            arr = input_generator(n)

            try:
                # Measure execution time
                start_time = time.time()
                sort_func(arr)
                elapsed_time = time.time() - start_time

                total_time += elapsed_time
            except RecursionError:
                print(
                    f"Warning: RecursionError for size {n}. Consider smaller sizes or increasing the recursion limit.")
                # Set a large time to indicate failure
                elapsed_time = float('inf')
                total_time = float('inf')
                break

        # Calculate average time
        if total_time == float('inf'):
            avg_time = float('inf')
            print(f"Size {n}: Failed due to recursion limit")
        else:
            avg_time = total_time / repeats
            print(f"Size {n}: {avg_time:.6f} seconds")

        results.append((n, avg_time))

    return results


# Run benchmarks and plot results
def run_benchmarks():
    """
    Run benchmarks for both quicksort implementations and plot results.
    """
    # Define input sizes to test - reduced for safety
    sizes = [100, 500, 1000, 1500, 2000, 2500, 3000]

    # Run benchmarks
    print("Running best case benchmarks...")
    best_case_results = benchmark(quicksort_nonrandom, generate_best_case, sizes)

    print("\nRunning worst case benchmarks...")
    worst_case_results = benchmark(quicksort_nonrandom, generate_worst_case, sizes)

    print("\nRunning average case benchmarks...")
    avg_case_results = benchmark(quicksort_nonrandom, generate_average_case, sizes)

    # Filter out any failed results
    best_case_results = [(size, time) for size, time in best_case_results if time != float('inf')]
    worst_case_results = [(size, time) for size, time in worst_case_results if time != float('inf')]
    avg_case_results = [(size, time) for size, time in avg_case_results if time != float('inf')]

    # Plot results
    plt.figure(figsize=(10, 6))

    # Extract x and y values for plotting
    if best_case_results:
        x_best, y_best = zip(*best_case_results)
        plt.plot(x_best, y_best, 'o-', label='Best Case (Median as Pivot)')

    if worst_case_results:
        x_worst, y_worst = zip(*worst_case_results)
        plt.plot(x_worst, y_worst, 'o-', label='Worst Case (Reverse Sorted)')

    if avg_case_results:
        x_avg, y_avg = zip(*avg_case_results)
        plt.plot(x_avg, y_avg, 'o-', label='Average Case (Random)')

    # Add reference lines for common complexities if we have data
    if best_case_results or worst_case_results or avg_case_results:
        x_values = []
        if best_case_results:
            x_values.extend(x_best)
        if worst_case_results:
            x_values.extend(x_worst)
        if avg_case_results:
            x_values.extend(x_avg)

        x = np.array(sorted(set(x_values)))
        plt.plot(x, 2e-7 * x * np.log(x), '--', label='O(n log n) reference')
        plt.plot(x, 1e-8 * x ** 2, '--', label='O(n²) reference')

    # Set labels and title
    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Quicksort Performance (Non-Random Pivot)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('quicksort_benchmark.png')
    plt.show()


# Compare both implementations
def compare_implementations():
    """
    Compare the performance of both quicksort implementations.
    """
    print("\nComparing quicksort implementations...")
    sizes = [1000, 2000, 3000, 4000, 5000]

    # Use average case (random arrays) for comparison
    random_results = benchmark(quicksort_random, generate_average_case, sizes)
    nonrandom_results = benchmark(quicksort_nonrandom, generate_average_case, sizes)

    # Filter out any failed results
    random_results = [(size, time) for size, time in random_results if time != float('inf')]
    nonrandom_results = [(size, time) for size, time in nonrandom_results if time != float('inf')]

    # Plot comparison
    plt.figure(figsize=(10, 6))

    if random_results:
        x_random, y_random = zip(*random_results)
        plt.plot(x_random, y_random, 'o-', label='Random Pivot')

    if nonrandom_results:
        x_nonrandom, y_nonrandom = zip(*nonrandom_results)
        plt.plot(x_nonrandom, y_nonrandom, 'o-', label='Non-Random Pivot')

    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Comparison of Quicksort Implementations')
    plt.legend()
    plt.grid(True)

    plt.savefig('quicksort_comparison.png')
    plt.show()


if __name__ == "__main__":
    # Test the implementations
    test_arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(f"Original array: {test_arr}")
    print(f"Non-random pivot sort: {quicksort_nonrandom(test_arr)}")
    print(f"Random pivot sort: {quicksort_random(test_arr)}")

    # Run benchmarks for non-random pivot
    run_benchmarks()

    # Compare both implementations
    compare_implementations()