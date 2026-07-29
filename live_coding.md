# SIMD Live Coding

### util.h

```hpp
#pragma once

#include <experimental/simd>
#include <iostream>
#include <cstddef>

/**
 * @brief Overloads the << operator to print a SIMD vector. Prints each element
 * of the SIMD vector with two spaces between elements.
 *
 * @tparam T   Element type of the SIMD vector (e.g., int, float)
 * @tparam Abi ABI tag that defines the layout of the SIMD vector
 * @param os   Output stream to print to
 * @param simd The SIMD vector to be printed
 * @return Reference to the modified output stream
 */
template <class T, class Abi>
std::ostream &operator<<(std::ostream &os,
                         const std::experimental::simd<T, Abi> &simd) {
  for (unsigned int i = 0; i < simd.size(); ++i) {
    os << simd[i] << "  ";
  }
  return os;
}

/**
 * @brief Prints the contents of a container with a label. The function is
 * templated to support any container that can be iterated using range-based for
 * loops (e.g., std::array, std::vector).
 *
 * @tparam container Type of the container
 * @param c          Container whose elements are to be printed
 * @param label      Label to print before the container's contents
 */
template <typename container>
void print_container(const container& c, const std::string& label)
{
  std::cout << label << ": [ ";
  for (const auto &e : c) {
    std::cout << e << " ";
  }
  std::cout << "]" << std::endl;
}
```


### simd_type_intro.cpp

```cpp
#include <experimental/simd> // Header for SIMD experimental features
#include <iostream>
#include <numeric> // For std::iota

#include "util.h" // Helper functions

// Alias for the experimental namespace
namespace stdx = std::experimental;

int main() {
  /******************************************************
   * General type definition using fixed and native SIMD ABI
   ******************************************************/
  // SIMD with 4 (fixed-size) double elements
  stdx::simd<double, stdx::simd_abi::fixed_size<4>> fixed{};
  // SIMD using the native size for `double` on the current
  // hardware
  stdx::simd<double, stdx::simd_abi::native<double>> native{};

  /******************************************************
   * Fixed size SIMD with different sizes
   ******************************************************/
  // 2 double elements (initialized with 0)
  stdx::fixed_size_simd<double, 2> fixed_2{};
  // 4 double elements (initialized with 0)
  stdx::fixed_size_simd<double, 4> fixed_4{};
  // 8 double elements (default construction, uninitialized or
  // zero-initialized depending on implementation)
  stdx::fixed_size_simd<double, 8> fixed_8;

  // Display the contents of each fixed-size SIMD
  std::cout << "Size 2: " << fixed_2 << std::endl;
  std::cout << "Size 4: " << fixed_4 << std::endl;
  std::cout << "Size 8: " << fixed_8 << std::endl;
  std::cout << std::endl;

  /******************************************************
   * Native SIMD
   ******************************************************/
  // SIMD with native width for `int`
  stdx::native_simd<int> my_native_simd;
  std::cout << "Constructed with native SIMD width: " << my_native_simd
            << std::endl;
  std::cout << std::endl;

  /******************************************************
   * Different ways of constructing SIMD values
   ******************************************************/
  // Default construction (uninitialized or zero-initialized depending on
  // implementation)
  stdx::native_simd<int> default_constructed;
  std::cout << "Default constructor: " << default_constructed << std::endl;

  // Initialize all lanes with value 73
  stdx::native_simd<int> initial_value(73);
  std::cout << "Initial value: " << initial_value << std::endl;

  // Initialize using a lambda to fill each lane with its index
  stdx::native_simd<int> generator([](int i) { return i; });
  std::cout << "Generator: " << generator << std::endl;

  std::cout << std::endl;

  /******************************************************
   * Loading and storing data from/to arrays
   ******************************************************/

  /* Unaligned memory case */

  // Regular std::array with no guaranteed alignment
  std::array<double, 10> unaligned_array{};
  // Fill with values 0 to 9
  std::iota(unaligned_array.begin(), unaligned_array.end(), 0);
  // Print initial values
  print_container(unaligned_array, "Unaligned before");

  // Native SIMD for double
  stdx::native_simd<double> simd;
  // Load elements from unaligned memory
  simd.copy_from(&unaligned_array[0], stdx::element_aligned);
  // Multiply each element by 2
  simd *= 2;
  // Store results back
  simd.copy_to(&unaligned_array[0], stdx::element_aligned);
  // Print modified values
  print_container(unaligned_array, "Unaligned after");

  /* Aligned memory case */
  // Ensure alignment for SIMD operations
  alignas(sizeof(double) * stdx::native_simd<double>::size())
      std::array<double, 10>
          aligned_array{};
  // Fill with 0 to 9
  std::iota(aligned_array.begin(), aligned_array.end(), 0);

  // Show original values
  print_container(aligned_array, "Aligned before");
  // Load using aligned memory access
  simd.copy_from(&aligned_array[0], stdx::vector_aligned);
  // Multiply each value by 100
  simd *= 100;
  // Store to offset location
  simd.copy_to(&aligned_array[2], stdx::vector_aligned);
  // Show modified values
  print_container(aligned_array, "Aligned after");

  return 0;
}
```

### simd_std_functions.cpp

```cpp
#include "util.h"
#include <array>             // For std::array
#include <experimental/simd> // Header for SIMD experimental features
#include <iostream>          // For std::cout

// Alias for the experimental namespace
namespace stdx = std::experimental;

/**
 * @brief Generates an array of N random integers in the range [0, 9].
 *
 * @tparam T Element type (e.g., int, float)
 * @tparam N Number of elements in the array
 * @return std::array<T, N> filled with random values
 */
template <typename T, int N> std::array<T, N> random_array() {
  std::array<T, N> arr{};
  for (auto &e : arr) {
    // Set to a random value between 0 and 9
    e = static_cast<T>(std::rand() % 10);
  }
  return arr;
}

int main() {
  /******************************************************
   * Splitting and Fusing SIMD Types
   ******************************************************/
  // Create a fixed-size SIMD of 32 uint16_t elements
  stdx::fixed_size_simd<uint16_t, 32> v;

  // Split the SIMD into four parts of sizes 4, 4, 8, and 16
  auto [split1, split2, split3, split4] = stdx::split<4, 4, 8, 16>(v);
  std::cout << "Sizes: " << split1.size() << ", " << split2.size() << ", "
            << split3.size() << ", " << split4.size() << std::endl;

  // Fuse the SIMD parts back into one full SIMD
  auto v_fuzed = stdx::concat(split1, split2, split3, split4);
  std::cout << "Fuzed size: " << v_fuzed.size() << std::endl;
  std::cout << std::endl;

  /******************************************************
   * Sorting Min and Max Values Between Two Arrays
   ******************************************************/
  constexpr auto arr_size = 5 * stdx::native_simd<int>::size();

  // Generate two random arrays of integers
  auto arr1 = random_array<int, arr_size>();
  auto arr2 = random_array<int, arr_size>();

  print_container(arr1, "Array 1 before");
  print_container(arr2, "Array 2 before");

  // For each SIMD chunk, compute the element-wise min and max
  for (unsigned int i = 0; i < arr_size; i += stdx::native_simd<int>::size()) {
    stdx::native_simd<int> x1, x2;
    // Load elements from the arrays
    x1.copy_from(&arr1[i], stdx::element_aligned);
    x2.copy_from(&arr2[i], stdx::element_aligned);

    // Compute element-wise min and max
    auto [min, max] = stdx::minmax(x1, x2);

    // Store min in arr1 and max in arr2
    min.copy_to(&arr1[i], stdx::element_aligned);
    max.copy_to(&arr2[i], stdx::element_aligned);
  }

  // Print final arrays to the console
  print_container(arr1, "Min Vector");
  print_container(arr2, "Max Vector");
  std::cout << std::endl;

  /******************************************************
   * SIMD Reduction Operation
   * Goal: Compute sum of all values in an array
   ******************************************************/
  std::array<int, arr_size> arr;
  arr.fill(1); // Fill with ones

  std::cout << "Array Size: " << arr_size << std::endl;

  // Initialize sum with zero. Used for accumulating array elements later
  int sum = 0;

  // SIMD reduction: accumulate all elements using stdx::reduce
  for (unsigned int i = 0; i < arr_size; i += stdx::native_simd<int>::size()) {
    // Load data to simd object
    stdx::native_simd<int> simd(&arr[i], stdx::element_aligned);
    // Reduction: Accumulate all elements in the simd object
    sum += stdx::reduce(simd, std::plus{});
  }

  // Print the final sum to the console
  std::cout << "Sum: " << sum << std::endl;

  return 0;
}
```

### simd_example_array_addition.cpp

```cpp
#include <array>
#include <experimental/simd> // Header for SIMD experimental features
#include <numeric>           // For std::iota

#include "util.h" // Helper functions

// Alias for the experimental namespace
namespace stdx = std::experimental;

int main() {
  /******************************************************
   * Define the involved vectors
   ******************************************************/
  // Input arrays `a` and `b` (size 13)
  std::array<int, 13> a;
  std::array<int, 13> b;

  // Output array `c`, aligned for SIMD operations
  alignas(stdx::memory_alignment_v<stdx::native_simd<int>>) std::array<int, 13>
      c;

  // Initialize `a` and `b` with values 0 to 12
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);

  // Display contents of `c` before the computation
  print_container(c, "Before");

  /******************************************************
   * Compute element-wise sum: c = a + b
   ******************************************************/
  std::size_t ub = c.size() - (c.size() % stdx::native_simd<int>::size());
  for (std::size_t i = 0; i < ub; i += stdx::native_simd<int>::size()) {
    // Load SIMD-width slice from `a` and `b`
    stdx::native_simd<int> x_simd(&a[i], stdx::element_aligned);
    stdx::native_simd<int> y_simd(&b[i], stdx::element_aligned);

    // Perform SIMD addition
    x_simd += y_simd;

    // Store result in aligned output array `c`
    x_simd.copy_to(&c[i], stdx::vector_aligned);
  }

  // dealing with the remainder -> In worksheet 1 you will explore the
  // vectorized version of the remaineder :)
  for (std::size_t i = ub; i < c.size(); ++i) {
    c[i] = a[i] + b[i];
  }

  // Display contents of `c` after the computation
  print_container(c, "After");

  return 0;
}
```

### simd_example_branching.cpp

```cpp
#include "util.h"
#include <cstdlib>           // For std::rand
#include <experimental/simd> // Header for SIMD experimental features
#include <vector>            // For std::vector

// Alias for the experimental namespace
namespace stdx = std::experimental;

/**
 * @brief Fills a vector with random integers in the range [min, max], with
 * randomly assigned signs.
 *
 * @param vec Reference to the vector to fill.
 * @param min Minimum absolute value for random numbers.
 * @param max Maximum absolute value for random numbers.
 */
void random_vector(std::vector<int> &vec, const int min, const int max) {
  for (int &val : vec) {
    int sign = (std::rand() % 2 == 0) ? 1 : -1;
    val = sign * (std::rand() % (max - min + 1) + min);
  }
}

int main() {
  /******************************************************
   * Generate a random aligned vector of integers
   ******************************************************/
  alignas(stdx::memory_alignment_v<stdx::native_simd<int>>) std::vector<int>
      vec(stdx::native_simd<int>::size() * 4, 0);

  // Fill the vector with random values between -9 and 9
  random_vector(vec, -9, 9);

  // Show the initial values
  print_container(vec, "Initial vector");

  /******************************************************
   * Perform vectorized computation:
   * - Multiply positive values by 2
   * - Replace negative values with 0
   ******************************************************/
  for (std::size_t i = 0; i < vec.size(); i += stdx::native_simd<int>::size()) {
    /* Initialize SIMD register with zero */
    stdx::native_simd<int> simd(0);
    stdx::native_simd_mask<int> mask;

    /* Step 1: Multiply values > 0 by two */
    // Load from vector
    simd.copy_from(&vec[i], stdx::vector_aligned);
    // Mask for positive values
    mask = simd > 0;
    // Multiply all by 2
    simd *= 2;
    // Store only positive results
    stdx::where(mask, simd).copy_to(&vec[i], stdx::vector_aligned);

    /* Step 2: Set values < 0 to zero */
    // Mask for negative values
    mask = simd < 0;
    // Prepare zero SIMD
    simd = 0;
    // Store zero simd only for negatives
    stdx::where(mask, simd).copy_to(&vec[i], stdx::vector_aligned);
  }

  // Show the modified vector
  print_container(vec, "Final vector");
}
```


# Thread Live Coding

### thread_basics.cpp



```cpp
#include <chrono>
#include <functional>
#include <iostream>
#include <syncstream>
#include <thread>

// Define a macro for thread-safe console output using std::osyncstream
#define sync_cout std::osyncstream(std::cout)

// A function to be run in a thread. Takes two integers, sleeps, and prints
// their sum.
void thread_function(int a, int b) {
  // Simulate some work by pausing this thread for 500 milliseconds
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  // Print the result in a thread-safe manner
  sync_cout << "[Function Thread] The sum of " << a << " and " << b << " is "
            << a + b << std::endl;
}

int main() {
  // Inform that threads are about to start
  sync_cout << "[Main Thread] Starting threads..." << std::endl;

  // Create a thread that runs the named function with 1 and 2 as arguments
  std::thread function_thread(thread_function, 1, 2);
  // Wait for thread to be done
  function_thread.join();

  // Local variables to be used by the lambda thread
  int a = 7;
  int b = 4;

  // Create a thread using a lambda that takes references to a and b
  // std::ref ensures that the actual variables a and b are passed by reference
  std::thread lambda_thread(
      [](int &a, const int &b) {
        sync_cout << "[Lambda Thread] The sum of " << a << " and " << b
                  << " is " << a + b << std::endl;
        a = 2;
      },
      std::ref(a),
      std::cref(
          b)); // std::ref is necessary to avoid copying and allow referencing

  // Join thread before printing a to the console
  lambda_thread.join();
  sync_cout << "[Main Thread] a = " << a << std::endl;

  // Create a std::jthread (automatically joins when it goes out of scope)
  {
    std::jthread jthread_thread(thread_function, 5, 6);
  }

  // Start a detached thread that runs the same function
  // This thread is not joined; it will run independently
  std::thread detached_thread(thread_function, 3, 4);
  detached_thread.detach();

  // Delay main to allow the detached thread time to finish. Without this, the
  // program might terminate before it completes
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Inform that all managed threads have finished
  sync_cout << "[Main Thread] All threads finished (except detached)."
            << std::endl;
  return 0;
}

```

### race_condition.cpp



```cpp
/**
 * @brief Demonstration of a simple race condition.
 *
 * This program creates three threads that increment a shared global counter
 * variable. Because the access to `counter` is not synchronized (i.e., no
 * locking or atomic operations), the three threads may interfere with each
 * other, leading to incorrect results — this is a race condition.
 */

#include <iostream>
#include <thread>

int main() {
  int counter = 0;

  /**
   * Lambda function that increments the shared counter 100,000 times.
   */
  auto count = [&]() -> void {
    for (int i = 0; i < 100000; ++i) {
      ++counter;
    }
  };

  // Launch two threads executing the count function in parallel
  std::thread thread1(count);
  std::thread thread2(count);
  std::thread thread3(count);

  // Wait for both threads to finish
  thread1.join();
  thread2.join();
  thread3.join();

  // Output the final value of the counter
  // Expected: 300000
  // Actual: Often less than 300000 due to race condition (you might need to run
  // this in debug to prevent the compiler from optimizing the loop away)
  std::cout << "Count: " << counter << std::endl;

  return 0;
}

```

### mutex_example.cpp



```cpp
/**
 * @brief Resolving race conditions with a std::mutex.
 *
 * This program demonstrates how to prevent race conditions using a mutex
 * (mutual exclusion). Three threads increment a shared counter in parallel, and
 * access to the counter is synchronized using a std::mutex to ensure that only
 * one thread can modify it at a time.
 */

#include <iostream>
#include <mutex>
#include <thread>

int main() {
  int counter = 0; // Shared counter variable
  std::mutex mtx;  // Mutex used to synchronize access to the counter

  /**
   * @brief Lambda function that increments the shared counter 100,000 times.
   *
   * The mutex is locked before modifying the counter and unlocked afterward.
   * This ensures mutual exclusion: only one thread at a time can increment the
   * counter.
   */
  auto count = [&]() -> void {
    for (int i = 0; i < 100000; ++i) {
      mtx.lock();   // Acquire the mutex (block if already locked)
      ++counter;    // Safely increment the shared counter
      mtx.unlock(); // Release the mutex
    }
  };

  // Create and start three threads that run the `count` lambda concurrently
  std::thread thread1(count);
  std::thread thread2(count);
  std::thread thread3(count);

  // Wait for all threads to complete
  thread1.join();
  thread2.join();
  thread3.join();

  // Output the final value of the counter
  // Expected: 3 * 100000 = 300000
  std::cout << "Count: " << counter << std::endl;

  return 0;
}

```

### lock_guard_example.cpp



```cpp
/**
 * @brief Using a lock guard.
 *
 * This program demonstrates how to use `std::lock_guard` to safely increment a
 * shared counter variable from multiple threads. The `std::lock_guard`
 * automatically acquires and releases the mutex, ensuring that the critical
 * section is properly protected and avoiding common mistakes such as forgetting
 * to unlock.
 */

#include <iostream>
#include <mutex>
#include <thread>

int main() {
  int counter = 0; // Shared counter to be incremented by all threads
  std::mutex lock; // Mutex used to synchronize access to `counter`

  /**
   * @brief Lambda function that increments the counter 100,000 times.
   *
   * A `std::lock_guard` is used to automatically manage the mutex. It locks the
   * mutex when created and unlocks it when it goes out of scope — here, at the
   * end of each loop iteration.
   */
  auto count = [&]() -> void {
    for (int i = 0; i < 100000; ++i) {
      std::lock_guard<std::mutex> local_lock(
          lock); // Locks the mutex for this scope
      ++counter; // Safely increment shared counter
      // Mutex is automatically released at the end of the loop iteration
    }
  };

  // Launch three threads to run the `count` function concurrently
  std::thread thread1(count);
  std::thread thread2(count);
  std::thread thread3(count);

  // Wait for all threads to complete
  thread1.join();
  thread2.join();
  thread3.join();

  // Print the final counter value
  // Expected: 3 * 100,000 = 300,000
  std::cout << "Count: " << counter << std::endl;

  return 0;
}

```

### deadlock_example.cpp



```cpp
/**
 * @brief Creating a deadlock.
 *
 * This program demonstrates a classic deadlock scenario. Two threads try to
 * acquire two mutexes (`mtx1` and `mtx2`) but in different orders. If both
 * threads acquire one mutex and then try to acquire the other, they can end up
 * waiting for each other indefinitely, causing a deadlock.
 */

#include <iostream>
#include <mutex>
#include <thread>

// Global counters to be incremented by threads
int counter1 = 0, counter2 = 0;

// Two mutexes to protect access to the counters
std::mutex mtx1, mtx2;

/**
 * @brief First thread function that locks mtx1 first, then mtx2.
 */
void count_thread1() {
  for (int i = 0; i < 100000; ++i) {
    mtx1.lock(); // Locks mtx1 first
    counter1++;

    mtx2.lock(); // Then tries to lock mtx2
    counter2++;
    mtx2.unlock();

    mtx1.unlock(); // Unlock mtx1
  }
}

/**
 * @brief Second thread function that locks mtx2 first, then mtx1.
 */
void count_thread2() {
  for (int i = 0; i < 100000; ++i) {
    mtx2.lock(); // Locks mtx2 first (opposite order)
    counter2++;

    mtx1.lock(); // Then tries to lock mtx1
    counter1++;
    mtx1.unlock();

    mtx2.unlock(); // Unlock mtx2
  }
}

int main() {
  // Launch both threads
  std::thread thread1(count_thread1);
  std::thread thread2(count_thread2);

  // Wait for both threads to complete
  thread1.join();
  thread2.join();

  // Print final counter values
  std::cout << "Count 1: " << counter1 << std::endl;
  std::cout << "Count 2: " << counter2 << std::endl;

  return 0;
}

```

### condition_variable_example.cpp



```cpp
/**
* @brief A basic example for using condition variables.
*
* This example demonstrates how a producer thread can signal a consumer thread
* using a condition variable once some shared data has been prepared.
*/

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <syncstream>
#include <thread>

int main() {
 // Shared flag to notify the consumer thread
 bool notify = false;

 // Shared data to be used by the consumer thread and set by the producer thread
 int shared_data = 0;

 // Mutex for synchronizing access to shared variables
 std::mutex mtx;

 // Condition variable for thread communication
 std::condition_variable cv;

 // Consumer thread lambda function
 auto consumer_task = [&]() {
   // Lock the mutex before accessing shared data
   std::unique_lock<std::mutex> lock(mtx);

   // Wait until `notify` becomes true; the lambda is a predicate
   // that will only allow the thread to continue when it returns true.
   cv.wait(lock, [&] { return notify; });

   // Output the shared data using thread-safe output
   std::osyncstream(std::cout)
       << "Received: The value of shared_data is: " << shared_data
       << std::endl;

   // The lock is released automatically when it goes out of scope
 };

 // Producer thread lambda function
 auto producer_task = [&]() {
   // Lock the mutex before modifying shared data
   mtx.lock();

   // Simulate data production
   shared_data = 4 + 8 + 15 + 16 + 23 + 42;

   // Set the notify flag to true so the consumer can proceed
   notify = true;

   // Unlock the mutex after modification
   mtx.unlock();

   // Notify one waiting thread (the consumer) that data is ready
   cv.notify_one();
 };

 // Launch consumer and producer threads
 std::jthread consumer_thread(consumer_task);
 std::jthread producer_thread(producer_task);

 // Main thread exits, jthreads handle their own joining
 return 0;
}

```

# OpenMP Live Coding

### 1_omp_nested.cpp



```cpp
#include <chrono>
#include <iostream>
#include <omp.h> // For OpenMP functions like omp_get_thread_num(), etc.
#include <string>
#include <thread>

int main() {
  // Set maximum nesting levels for parallel regions.
  // Allows OpenMP to create nested parallel regions up to 2 levels deep.
  omp_set_max_active_levels(2);

// First-level parallel region with 2 threads
#pragma omp parallel num_threads(2)
  {
    // Get the thread ID for the outer parallel region
    const int id = omp_get_thread_num();

// Nested (second-level) parallel region, also with 2 threads
#pragma omp parallel num_threads(2)
    {
      // Only the nested threads created by thread 0 of the outer region
      // execute this condition
      if (id == 0) {
        // Simulate some work with 1-second sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // Output to indicate this is a nested thread from thread 0
        std::cout << "This is a thread spawned by thread 0!\n";
      }
    }

    // This string shows the end of the outer thread's nested parallel
    // section
    std::string output = "At the end of the parallel region of thread " +
                         std::to_string(id) + "\n";

    // Print the output message
    std::cout << output;

// Synchronize the two outer threads to wait until both reach this point
#pragma omp barrier

    // After synchronization, each outer thread prints this message
    std::cout << "Manually synchronized!\n";
  }

  // After all threads finish, print the final message from the main
  // thread
  std::cout << "At the end of the program!" << std::endl;

  return 0;
}

```

### 2_omp_critical.cpp



```cpp
#include <iostream>

int main() {
  int counter = 0;         // Shared counter across all threads
  const int n_threads = 3; // Number of threads to use in the parallel region

  // Start of OpenMP parallel region using 3 threads
#pragma omp parallel num_threads(n_threads)
  {
    // Each thread executes this loop 100,000 times!
    for (int i = 0; i < 100'000; ++i)
#pragma omp critical // Ensure only one thread modifies 'counter' at a time
      ++counter;     // Safely increment shared counter
  }

  // Print the final value of the counter after all threads finish
  std::cout << "Count: " << counter << std::endl;

  return 0;
}

```

### 3_omp_single.cpp



```cpp
#include <chrono>     // For measuring/managing time intervals
#include <iostream>
#include <omp.h>
#include <syncstream> // For synchronized output to std::cout in multithreading
#include <thread>     // For thread sleep and timing utilities

// Define a macro for synchronized output stream using
#define sync_cout std::osyncstream(std::cout)

int main() {
// Start of OpenMP parallel region with 4 threads
#pragma omp parallel num_threads(4)
  {
    // Each thread gets its own unique ID (from 0 to 3)
    int thread_id = omp_get_thread_num();

    sync_cout << "Hello from thread #" << thread_id << "\n";

// Only a single thread will execute the following block (others wait)
#pragma omp single
    {
      // Identify which thread is executing the single section
      sync_cout << "Thread #" << thread_id << " is the chosen one!\n";

      // Simulate an expensive computation or workload with 1 second delay
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));

      sync_cout << "End of the single section.\n";
    }

    // All threads will execute this line after the single block
    sync_cout << "Thread #" << thread_id << " continues the journey!\n";
  } // implicit barrier

  return 0;
}

```

### 4_omp_manual_worksharing.cpp



```cpp
#include <numeric>
#include <omp.h>
#include <vector>
#include <iostream>

int main() {
  constexpr std::size_t n = 777;        // Size of the vectors
  std::vector<double> a(n), b(n), c(n); // Input vectors a, b; output vector c

  // Initialize both 'a' and 'b' with values 0 to n-1
  for (auto *vec : {&a, &b})
    std::iota(vec->begin(), vec->end(), 0); // Fill with 0, 1, 2, ..., n-1

#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();  // Each thread gets its own ID
    const int n_threads = omp_get_num_threads(); // Total number of threads

    // Calculate the range of indices this thread should process
    const int start_index = thread_id * n / n_threads;
    const int end_index =
        (thread_id == n_threads - 1) ? n : (thread_id + 1) * n / n_threads;

    // Each thread performs addition on its assigned chunk
    for (int i = start_index; i < end_index; ++i)
      c[i] = a[i] + b[i];
  }

  // Print the result of the addition to the console
  std::cout << "[ ";
  for (const auto elem : c)
    std::cout << elem << " ";
  std::cout << "]" << std::endl;

  return 0;
}

```

### 5_omp_for_reduction.cpp



```cpp
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <limits>

int main() {
  constexpr std::size_t n = 777;
  std::vector<double> a(n), b(n), c(n);

  // Initialize vectors a and b with 0, 1, 2, ..., n-1
  for (auto *vec : {&a, &b})
    std::iota(vec->begin(), vec->end(), 0);

  // Reduction variables
  double max_value, min_value;

#pragma omp parallel
  {
    // Compute element-wise sum: c[i] = a[i] + b[i]
#pragma omp for schedule(static, 73)
    for (std::size_t i = 0; i < n; ++i) {
      c[i] = a[i] + b[i];
    }

    // Find max of c using reduction
#pragma omp for reduction(max : max_value) schedule(dynamic, 10) nowait
    for (std::size_t i = 0; i < n; ++i)
      max_value = std::max(max_value, c[i]);

    // Find min of c using reduction
#pragma omp for reduction(min : min_value) schedule(guided, 10)
    for (std::size_t i = 0; i < n; ++i)
      min_value = std::min(min_value, c[i]);
  }

  // Output results
  std::cout << "Max: " << max_value << std::endl;
  std::cout << "Min: " << min_value << std::endl;

  return 0;
}

```

### 6_omp_collapse.cpp



```cpp
#include <iostream>
#include <omp.h>
#include <syncstream> // For synchronized output to std::cout in multithreading

// Macro to use synchronized output with std::cout
#define sync_cout std::osyncstream(std::cout)

int main() {
  // Parallel for loop with 2 threads
  // collapse(2): merges two nested loops (i, j) into a single loop of 6
  //              iterations (2 * 3)
  // schedule(static, 1): distributes one iteration at a time  to each thread in
  //                      round-robin fashion
#pragma omp parallel for num_threads(2) schedule(static, 1) collapse(2)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      sync_cout << "This is thread " << omp_get_thread_num()
                << " working on iter (i,j) = (" << i << ", " << j << ")\n";
    }

  return 0;
}

```

### 7_omp_ordered.cpp



```cpp
#include <chrono> // For measuring/managing time intervals
#include <iostream>
#include <omp.h>
#include <syncstream> // For synchronized output to std::cout in multithreading
#include <thread>     // For thread sleep and timing utilities

// Macro to use synchronized output with std::cout
#define sync_cout std::osyncstream(std::cout)

int main() {
// Start a parallel region with 4 threads
// Parallel loop with ordering enforced for certain operations
#pragma omp parallel for ordered num_threads(4)
    for (int i = 0; i < 8; ++i) {
      sync_cout << "Horse " << i << " is galloping toward the bridge (thread "
                << omp_get_thread_num() << ")\n";

      // Simulate time taken for the horse to reach the bridge
      std::this_thread::sleep_for(std::chrono::seconds(1));

// Ensure the bridge-crossing section executes in order of i (sequentially)
#pragma omp ordered
      {
        // Only one horse can cross at a time, and in order
        sync_cout << "Horse " << i << " is crossing the bridge...\n";

        // Simulate crossing time
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        // Confirm the horse has crossed
        sync_cout << "Horse " << i << " has crossed the bridge.\n";
      } // no barrier here
    }

  return 0;
}

```

### 8_omp_data_sharing.cpp



```cpp
#include <iostream>
#include <omp.h>
#include <syncstream> // For synchronized output to std::cout in multithreading

// Macro to use synchronized output with std::cout
#define sync_cout std::osyncstream(std::cout)

int main() {
  // Variable definitions
  int shared_var = 73;   // Will be shared across all threads
  int private_var = 777; // Will be private to each thread (uninitialized inside
                         // parallel region)
  long firstprivate_var =
      108; // Will be private, but initialized with the value from outside
  int lastprivate_var = 747; // Will be private, but the value from the last
                             // loop iteration will be written back

  constexpr int n_threads = 4;

  // Print the initial state before entering the parallel region
  sync_cout << "--------- Before the parallel region ---------\n";
  sync_cout << "Shared variable:       " << shared_var << "\n";
  sync_cout << "Private variable:      " << private_var << "\n";
  sync_cout << "Firstprivate variable: " << firstprivate_var << "\n";
  sync_cout << "Lastprivate variable:  " << lastprivate_var << "\n";

// Start of parallel region with 4 threads
#pragma omp parallel num_threads(n_threads) default(none)                      \
    shared(shared_var, lastprivate_var, std::cout) private(private_var)        \
    firstprivate(firstprivate_var)
  {
// Parallel for loop — each thread gets iterations of this loop
// The value of 'lastprivate_var' from the last logical iteration is saved to
// the original after the loop
#pragma omp for lastprivate(lastprivate_var)
    for (int i = 0; i < n_threads; ++i) {
// Critical section ensures only one thread executes this block at a time
#pragma omp critical
      {
        shared_var += i;      // shared_var is visible to all threads
        private_var = i * 10; // each thread has its own copy of private_var
        firstprivate_var +=
            i; // initialized to 108, then modified independently
        lastprivate_var =
            i * 100; // value from last loop iteration will be retained
      }
    }

    // Print each thread's view of the variables
    sync_cout << "Thread " << omp_get_thread_num() << "\n";
    sync_cout << "  Shared variable:       " << shared_var << "\n";
    sync_cout << "  Private variable:      " << private_var << "\n";
    sync_cout << "  Firstprivate variable: " << firstprivate_var << "\n";
    sync_cout << "  Lastprivate variable:  " << lastprivate_var << "\n";
  }

  // Print final state after the parallel region ends
  sync_cout << "--------- After the parallel region ---------" << "\n";
  sync_cout << "Shared variable:       " << shared_var << "\n";
  sync_cout << "Private variable:      " << private_var
            << "   (unchanged outside region)\n";
  sync_cout << "Firstprivate variable: " << firstprivate_var
            << "   (unchanged outside region)\n";
  sync_cout << "Lastprivate variable:  " << lastprivate_var
            << "   (updated from last iteration)\n";

  return 0;
}

```


# MPI Live Coding

### 1_mpi_basics.cpp



```cpp
#include <iostream>
#include <mpi.h>
#include <string>

int main() {
  // Initialize a greeting message string
  std::string greeting = "Hello, world!";

  // Variables to hold the MPI process rank and total number of processes
  int rank, size;

  // Initialize the MPI environment. Must be called before any other MPI
  // functions.
  MPI_Init(nullptr, nullptr);

  // Get the rank of the current process within MPI_COMM_WORLD communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the total number of processes in the MPI_COMM_WORLD communicator
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Create a message string
  std::string msg = "This is rank " + std::to_string(rank) + " of " +
                    std::to_string(size) + " saying: " + greeting + "\n";

  // Print the message to standard output
  std::cout << msg;

  // Finalize the MPI environment. No MPI calls allowed after this.
  MPI_Finalize();

  return 0;
}

```

### 2_mpi_chain.cpp



```cpp
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
  // Initialize the MPI environment with command-line arguments
  MPI_Init(&argc, &argv);

  int rank;
  int size;
  // Get the current process's rank in the MPI_COMM_WORLD communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Get the total number of processes in MPI_COMM_WORLD
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Ensure that there are at least 2 processes to run the chain of communication
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2)
    MPI_Abort(MPI_COMM_WORLD, 73);

  int tag = 0; // Message tag for matching sends and receives
  int data;    // Integer data that will be passed along the chain

  if (rank == 0) {
    // Rank 0 starts the chain by initializing data to 0
    data = 0;
    std::cout << "[Rank " << rank << "] Starting chain with value: " << data
              << std::endl;
    // Send the initial data to the next rank (rank 1)
    MPI_Send(&data, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD);
  } else if (rank == size - 1) {
    // The last rank receives the data from the previous rank
    MPI_Recv(&data, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // Increment the received value by 1
    data += 1;
    // Print the final value after all increments
    std::cout << "[Rank " << rank << "] Final value after passing: " << data
              << std::endl;
  } else {
    // All other middle ranks receive from previous rank...
    MPI_Recv(&data, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // Increment the received value by 1
    data += 1;
    // Print the updated value and the next rank to send to
    std::cout << "[Rank " << rank << "] Received, incremented to " << data
              << ", passing to Rank " << rank + 1 << std::endl;
    // Send the incremented value to the next rank
    MPI_Send(&data, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD);
  }

  // Finalize the MPI environment; no MPI calls after this
  MPI_Finalize();
  return 0;
}

```

### 3_mpi_non_blocking.cpp



```cpp
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <thread>

int main(int argc, char **argv) {
  // Initialize MPI environment with command-line arguments
  MPI_Init(&argc, &argv);

  int rank;
  // Get the rank (ID) of the current process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2)
    MPI_Abort(MPI_COMM_WORLD, 73);

  const int tag = 73; // Message tag for matching send and receive
  int data;           // Data to be sent or received

  MPI_Request request; // MPI request handle for non-blocking operations
  MPI_Status status;   // MPI status object to get message info

  if (rank == 0) {
    // Data to send
    data = 747;
    // Simulate some work with a half-second delay before sending
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Non-blocking send: Initiates send operation and returns immediately
    MPI_Isend(&data, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &request);

    // Wait explicitly for the non-blocking send to complete before proceeding
    MPI_Wait(&request, &status);
    std::cout << "[Rank 0] Data sent: " << data << std::endl;

  } else if (rank == 1) {
    data = 0; // Initialize receive buffer
    bool received = false;

    // Non-blocking receive: Initiates receive operation and returns immediately
    MPI_Irecv(&data, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &request);

    // Poll in a loop to check if the message has arrived without blocking
    while (!received) {
      int flag = 0;
      // Test if the non-blocking receive has completed
      MPI_Test(&request, &flag, &status);
      if (flag) {
        // Message received, set flag to exit loop and print data
        received = true;
        std::cout << "[Rank 1] Data received (via MPI_Test): " << data
                  << std::endl;
      } else {
        // Message not yet received, simulate doing other work before retrying
        std::cout << "[Rank 1] Waiting for data..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    }

    // Demonstrate MPI_Wait here as well — this will return immediately since
    // the receive is already complete.
    MPI_Wait(&request, &status);
    std::cout << "[Rank 1] Final confirmation of receive (via MPI_Wait)."
              << std::endl;
  }

  // Finalize the MPI environment
  MPI_Finalize();
  return 0;
}

```