import random
import time
import matplotlib.pyplot as plt
import numpy as np

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def quicksort_random(arr):
    if len(arr) <= 1:
        return arr
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_random(left) + middle + quicksort_random(right)

def benchmark_quicksort(sort_func, input_generator, sizes):
    times = []
    for n in sizes:
        arr = input_generator(n)
        start = time.time()
        sort_func(arr)
        end = time.time()
        times.append(end - start)
    return times

def best_case(n):
    return list(range(n))  # Already sorted

def worst_case(n):
    return list(range(n, 0, -1))  # Reverse sorted

def average_case(n):
    return [random.randint(0, n) for _ in range(n)]

sizes = [100, 500, 1000, 5000, 10000]
best_times = benchmark_quicksort(quicksort, best_case, sizes)
worst_times = benchmark_quicksort(quicksort, worst_case, sizes)
avg_times = benchmark_quicksort(quicksort, average_case, sizes)

plt.figure(figsize=(10, 5))
plt.plot(sizes, best_times, label='Best Case', marker='o')
plt.plot(sizes, worst_times, label='Worst Case', marker='o')
plt.plot(sizes, avg_times, label='Average Case', marker='o')
plt.xlabel('Input Size (n)')
plt.ylabel('Time (s)')
plt.title('Quicksort Performance')
plt.legend()
plt.grid()
plt.show()
