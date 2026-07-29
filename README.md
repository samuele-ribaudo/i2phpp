# SIMD

## 1. Fundamentals of SIMD Architecture

* **Data-Level Parallelism:** Single Instruction, Multiple Data (SIMD) processes multiple data elements simultaneously using a single instruction executed across vector registers.

* **Flynn's Taxonomy:** SIMD falls under Flynn's classification alongside SISD (sequential), MISD, and MIMD (multicore/distributed).

* **Native vs. Fixed-Size Registers:**
* **Native ABI (`stdx::native_simd<T>`):** Scales automatically to the vector register width supported by the host CPU (e.g., 4 lanes for standard 32-bit integers on 128-bit ARM NEON/AVX registers).

* **Fixed-Size ABI (`stdx::fixed_size_simd<T, N>`):** Explicitly enforces an N-element vector length across any compilation target.

---

## 2. Construction & Initialization

* **Broadcast Initialization:** Initializing a vector with a single scalar assigns that value across all lanes simultaneously (e.g., `stdx::native_simd<int> simd(73)` fills all lanes with `73`).

* **Generator Constructor:** Initializing via a lambda function calls the lambda once per lane, passing the lane's index i as an argument (e.g., `[](int i) { return i; }` produces `[0, 1, 2, 3]`).

---

## 3. Data Transfer & Memory Alignment

* **Element-Aligned Access (`stdx::element_aligned`):** Used when loading or storing data from standard, unaligned memory boundaries (e.g., standard `std::array` or `std::vector`).

* **Vector-Aligned Access (`stdx::vector_aligned`):** Requires memory to be aligned to vector boundaries using `alignas(...)` with `stdx::memory_alignment_v<SimdType>`. Enforces hardware-level aligned load/store instructions for optimal CPU execution speed.

* **Member Functions:** `copy_from(ptr, alignment_tag)` loads memory into CPU registers, and `copy_to(ptr, alignment_tag)` writes SIMD register contents back to memory.

---

## 4. Key SIMD Utility & Standard Functions

### Splitting and Fusing Operations

* **`stdx::split<N1, ... N2,>(v)`:** Decomposes a larger SIMD register into structured bindings of smaller SIMD vectors.

* **`stdx::concat(v1, v2, ...)`:** Combines multiple smaller SIMD vectors into a single wider register.

### Comparisons & Reductions

* **`stdx::minmax(x1, x2)`:** Performs element-wise lane comparisons between two vectors simultaneously and returns a structured tuple `[min, max]` containing the element-wise minimums and maximums.

* **`stdx::reduce(simd, binary_op)`:** Performs horizontal reduction across all lanes within a vector register (e.g., combining lanes using `std::plus{}` to sum all internal elements).

---

## 5. Branching & Masking Logic

* **Divergence Handling:** SIMD hardware cannot execute standard `if/else` conditional jumps per element without breaking parallel execution.

* **Mask Vectors (`stdx::native_simd_mask<T>`):** Storing conditional expressions (e.g., `mask = simd > 0`) creates a boolean mask vector.

* **Conditional Writes (`stdx::where`):** Standard conditional updates write results back to selective lanes using masked predicates:
```cpp
// Multiplies vector values by 2 and writes back ONLY where mask is true
stdx::where(mask, simd).copy_to(&vec[i], stdx::vector_aligned);

```
---

# WORKSHEET 1 BREAKDOWN

## 1. Task 1: Vector Class Vectorization (`LaVector<number, N>`)

### The Class Template Signature

```cpp
template <typename number, int N = 1>
class LaVector;

```

* **`number`**: Primitive data type (`float`, `double`, `int`).

* **`N`**: Vectorization width (number of lanes per SIMD register). Default `1` preserves scalar fallback.

---

### Implementation Pattern: Main Loop + Masked Remainder

In `la_vector.hpp`, vector operations (`operator+=`, `operator-=`, `operator*=`, `operator/=`, `schur_product`) follow a standardized SIMD pattern:

```cpp
// 1. Calculate Upper Bound and Remainder
const std::size_t ub        = this->size() - (this->size() % N);
const std::size_t remainder = this->size() % N;

// 2. Vectorized Main Loop
for (std::size_t i = 0; i < ub; i += N) {
    stdx::fixed_size_simd<number, N> x_simd(&data[i], stdx::element_aligned);
    stdx::fixed_size_simd<number, N> y_simd(&vec.data[i], stdx::element_aligned);
    x_simd += y_simd;
    x_simd.copy_to(&data[i], stdx::element_aligned);
}

// 3. Masked SIMD Remainder Loop
if (remainder > 0) {
    stdx::fixed_size_simd_mask<number, N> mask(false);
    for (std::size_t i = 0; i < remainder; ++i)
        mask[i] = true;

    stdx::fixed_size_simd<number, N> x_simd(0);
    stdx::fixed_size_simd<number, N> y_simd(0);

    stdx::where(mask, x_simd).copy_from(&data[ub], stdx::element_aligned);
    stdx::where(mask, y_simd).copy_from(&vec.data[ub], stdx::element_aligned);

    x_simd += y_simd;

    stdx::where(mask, x_simd).copy_to(&data[ub], stdx::element_aligned);
}

```

#### Why did you implement it this way?

1. **`stdx::element_aligned` vs `stdx::vector_aligned`**: The underlying container is `std::vector<number>`. Since `std::vector` allocations on the heap are not guaranteed to be aligned to SIMD register boundaries (e.g., 32-byte or 64-byte alignment), using `stdx::element_aligned` prevents unaligned memory access faults.

2. **Masked Remainder Vectorization (`stdx::where`)**: Instead of a scalar fallback loop for the leftover elements (`size % N`), you used `stdx::fixed_size_simd_mask`. `stdx::where(mask, ...)` conditionally loads and writes back only the active lanes where the mask is `true`, leaving out-of-bound memory untouched.

---

### Reductions & Norms

For operations like `l2_norm_squared()`, `norm_l1()`, and `norm_linf()`:

* You accumulate values inside a single SIMD vector register (`acc_simd`) across the entire loop.

* At the end, you collapse the vector lanes into a single scalar value using **horizontal reduction** (`stdx::reduce` or `stdx::hmax`):

```cpp
// Horizontal sum across SIMD vector lanes
return stdx::reduce(acc_simd, std::plus<>()); 

```

---

### Oral Exam Defense Questions (Task 1)

#### Q1: Why can an incorrect loop bound (reading beyond array size) work without crashing during testing? Why is it unsafe?

* **Why it doesn't crash**: Memory is allocated in virtual pages (typically 4 KB) by the OS, and `std::vector` allocates heap capacity in chunks. If an out-of-bounds SIMD load accesses bytes that lie within the same memory page or allocated capacity, no hardware fault occurs.

* **Why it's dangerous**:
1. **Undefined Behavior (UB)**: Reading uninitialized or out-of-bounds data.

2. **Page Faults / Crashes**: If a vector ends near a page boundary, a 128-bit/256-bit SIMD load that crosses into an unmapped page will trigger a **Segmentation Fault (`SIGSEGV`)**.

3. **Data Pollution**: If masked stores are not used, invalid data gets written back to memory.

---

## 2. Task 2: Matrix Class Vectorization (`LaMatrix<number, N>`)

### Internal Storage Layout

```cpp
std::vector<std::vector<number>> data;

```

* **Memory Structure**: A vector of dynamic vectors.

* **Layout Implication**: Each individual row is contiguous in memory. However, **different rows are separate heap allocations and NOT contiguous with each other**.

---

### A. Element-Wise Operations & Matrix-Vector Multiplication

For Matrix-Vector product ($y = A \cdot x$):

```cpp
for (std::size_t row_idx = 0; row_idx < rows(); ++row_idx) {
    std::size_t idx = 0;
    number sum(0.0);
    for (; idx + simd_width <= cols(); idx += simd_width) {
        simd_type row_vec(&data[row_idx][idx], stdx::element_aligned);
        simd_type vec_elements(&vec[idx], stdx::element_aligned);
        simd_type product = row_vec * vec_elements;
        sum += stdx::reduce(product);
    }
    // Scalar loop for column remainder
    for (; idx < cols(); ++idx) {
        sum += (*this)[row_idx, idx] * vec[idx];
    }
    result[row_idx] = sum;
}

```

* **Explanation**: You iterate through matrix rows. Since row elements $A_{i, j \dots j+N}$ are contiguous in memory, you perform element-wise SIMD multiplication with vector chunk $x_{j \dots j+N}$ and horizontally sum the results into a scalar accumulator.

---

### B. Matrix-Matrix Product Optimization ($C = A \cdot B$)

Standard matrix multiplication ($C_{i, j} = \sum_k A_{i, k} B_{k, j}$) processes matrix $B$ column-wise. Because `LaMatrix` stores rows independently, column accesses in $B$ are non-contiguous, leading to severe cache miss penalties.

#### Your Implementation (Loop Reordering $i \to k \to j$):

```cpp
for (std::size_t row_idx = 0; row_idx < rows(); ++row_idx)          // i
    for (std::size_t idx = 0; idx < cols(); ++idx) {                // k
        number a_scalar = (*this)[row_idx, idx];                    // A[i][k]
        std::size_t col_idx = 0;
        for (; col_idx + simd_width <= mat.cols(); col_idx += simd_width) { // j
            simd_type b_vec(&mat.data[idx][col_idx], stdx::element_aligned);
            simd_type res_vec(&result.data[row_idx][col_idx], stdx::element_aligned);
            
            res_vec += a_scalar * b_vec;
            
            res_vec.copy_to(&result.data[row_idx][col_idx], stdx::element_aligned);
        }
        // remainder processing...
    }

```

#### Why did you implement it this way?

1. **Scalar Broadcasting**: You pick a single scalar value $A_{i, k}$.

2. **Contiguous Access on Matrix B**: By vectorizing the innermost loop over $j$ (columns of $B$), you read row $k$ of matrix $B$ (`mat.data[idx][col_idx]`) as a **contiguous chunk** in memory.

3. **Cache Efficiency**: Every vectorized load from matrix $B$ is linear and fully utilizes cache lines.

---

### Oral Exam Defense Questions (Task 2)

#### Q1: Why is `trace()` computation NOT vectorized in your code?

* **Definition**: $\text{Trace}(A) = \sum_{i=0}^{M-1} A_{i, i}$.

* **Reason**: Diagonal elements $A_{0,0}, A_{1,1}, A_{2,2}, \dots$ are separated by a stride of $(\text{Cols} + 1)$ elements in continuous storage, or reside in completely disjoint `std::vector` allocations per row.

* **Conclusion**: SIMD registers require contiguous memory blocks for efficient loads. Gathering non-contiguous diagonal elements requires strided loads or gather instructions (`vpgatherd`), which incur higher latency than standard scalar addition.

#### Q2: Bonus Question — Cache Line Size vs. Vectorization Width Trade-off in GEMM?

* **Favorable Configuration**: When the total width of a SIMD register matches or cleanly divides the CPU **cache line size** (typically 64 bytes). For example, a 512-bit register (64 bytes = 8 `double`s) fills exactly one cache line per load operation.

* **Unfavorable Configuration**:
* If vector width exceeds cache line boundaries without alignment, memory loads split across multiple cache lines, causing cache lock stalls.

* If matrices $A, B, C$ are very large and do not fit in L1/L2 cache, row-reuse strategies break down unless combined with **Cache Tiling / Blocking** techniques.

---

## 3. Task 3: The Heat Operator & Solver (`HeatOperator`)

In `heat_operator.hpp`, the explicit Euler time step advances as:

$$\mathbf{T}^{(n+1)} = \mathbf{T}^{(n)} + \Delta t \cdot \left( \mathbf{A} \mathbf{T}^{(n)} + \mathbf{Q}^{(n)} \right)$$

```cpp
// Matrix-based Explicit Euler advancement
current_solution +=
    system_matrix * current_solution * current_time_increment +
    source_term_vector(current_time - current_time_increment) *
        current_time_increment;

```

### Key Connections to SIMD

1. `system_matrix * current_solution`: Triggers the SIMD-vectorized **Matrix-Vector product** (`LaMatrix::operator*`).

2. `current_solution += ...`: Triggers SIMD-vectorized **Vector addition and scalar scaling** (`LaVector::operator+=`).

3. Instantiation in `heat_solver.hpp`:

```cpp
using native_simd = std::experimental::simd<number, std::experimental::simd_abi::native<number>>;

HeatOperator<dim, number, LaVector<number, native_simd::size()>, LaMatrix<number, native_simd::size()>>

```

* The solver dynamically detects the native hardware SIMD lane width (`native_simd::size()`) for the target system.

---

## 4. Summary Table for Oral Exam Defense

| Concept / Method | Implementation Details | Key Theory / Reason |
| --- | --- | --- |
| **`LaVector` Remainder** | `stdx::fixed_size_simd_mask` + `stdx::where`<br> | Avoids scalar loops; prevents out-of-bound memory access safely.|
| **Memory Alignment** | `stdx::element_aligned`<br> | Data inside `std::vector` heap memory is not guaranteed vector-aligned.|
| **Matrix Storage** | `std::vector<std::vector<T>>`<br> | Rows are contiguous individually, but non-contiguous relative to each other.|
| **Matrix-Matrix (A*B) | i to k to j loop structure | Reads matrix B along contiguous row entries, eliminating strided column loads.|
| **`trace()` Vectorization** | Omitted (Scalar loop used)| Diagonal elements are non-contiguous; strided SIMD gathering incurs high overhead.|
| **Performance Limits** | Bandwidth Bottleneck (Memory Wall) | Matrix-vector multiplications are memory-bound; speedup plateaus at wide SIMD widths.|

---

# THREADS


## I. Theoretical Concepts & Architecture Takeaways

### 1. Architectural Classification

* **Flynn’s Taxonomy:** Multithreading falls under **MIMD** (Multiple Instruction, Multiple Data).
* **Memory Model:** Operates on **Shared Memory Systems**. All threads within a process share the same global address space.

### 2. Program vs. Process vs. Thread

* **Program:** A passive set of instructions stored on disk.
* **Process:** An executing instance of a program. Has its own isolated memory address space (Code, Data, Heap, File descriptors). Inter-process communication (IPC) is heavy and slow.
* **Thread:** The smallest lightweight unit of execution within a process.

| Component | Shared Across Threads in Process | Private to Each Thread |
| --- | --- | --- |
| **Memory / State** | Heap, Code Segment, Data Segment, Open Files | **Stack**, **Registers**, **Program Counter (PC)** |

### 3. Thread Scheduling & Lifecycle

* **Scheduling:** Managed by the OS Kernel. Uses context switches to assign CPU time slices to threads in the *Ready Queue*.
* **Lifecycle:**
1. **Creation:** Memory/stack allocated.
2. **Execution:** Scheduled and executed on CPU cores.
3. **Joining/Termination:** Waiting for completion and releasing thread resources.



---

## II. Live-Coding Code Syntax & API Reference

### 1. Basic Thread Management (`thread_basics.cpp`)

* **Header:** `#include <thread>`
* **Thread Creation & Joining:**
```cpp
std::thread t(func_name, arg1, arg2); // Launches thread immediately
t.join();                            // Main thread blocks until 't' completes

```


* **Passing References (`std::ref` / `std::cref`):**
`std::thread` constructors copy/move arguments by default. To pass variables by reference or const reference, wrap them in `std::ref()` or `std::cref()`:
```cpp
std::thread t(lambda_expr, std::ref(var1), std::cref(var2));

```


* **`std::jthread` (C++20):**
RAII-compliant thread wrapper. Automatically calls `.join()` on destruction (going out of scope), preventing resource leaks or process termination:
```cpp
{
    std::jthread jt(func_name, arg1, arg2); 
} // Automatically joined here

```


* **Detaching:**
```cpp
t.detach(); // Separates execution from object; runs independently in background.

```



---

### 2. Race Conditions & Synchronization (`race_condition.cpp`, `mutex_example.cpp`, `lock_guard_example.cpp`)

* **Race Condition:** Occurs when multiple threads access shared memory concurrently without synchronization, and at least one access is a write. Results in non-deterministic, corrupted state.
* **Manual Mutex (`std::mutex`):**
```cpp
#include <mutex>
std::mutex mtx;

mtx.lock();   // Critical Section (Only 1 thread at a time)
++counter;
mtx.unlock(); 

```


*Oral Exam Pitfall:* Manual `.lock()`/`.unlock()` is **not exception-safe**. If an exception occurs inside the critical section, the mutex remains locked forever (deadlock).
* **RAII Mutex Management (`std::lock_guard`):**
Locks on construction, automatically unlocks on destruction (exit scope / exception). Always prefer this over raw lock/unlock:
```cpp
{
    std::lock_guard<std::mutex> lock(mtx); // Locks mtx
    ++counter;                             // Critical section
} // Unlocks automatically here

```



---

### 3. Deadlocks (`deadlock_example.cpp`)

* **Definition:** A scenario where two or more threads are permanently blocked, each waiting for a mutex held by the other.
* **Cause in Code:** Lock acquisition in mismatched order across different threads:
* Thread 1 locks `mtx1` $\rightarrow$ requests `mtx2`
* Thread 2 locks `mtx2` $\rightarrow$ requests `mtx1`


* **Oral Exam Fix:** Always acquire locks in the **exact same linear order** across all threads (or use `std::scoped_lock` in C++17 to acquire multiple locks atomically).

---

### 4. Thread Communication (`condition_variable_example.cpp`)

* **Header:** `#include <condition_variable>`
* **Concept:** Allows threads to sleep until notified by another thread that a condition/data is ready.
* **Key Syntax:**
```cpp
std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// --- Consumer Thread ---
std::unique_lock<std::mutex> lock(mtx); // cv requires unique_lock (can lock/unlock dynamically)
cv.wait(lock, [&] { return ready; });  // Releases lock & sleeps until notified AND ready==true

// --- Producer Thread ---
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}
cv.notify_one(); // Unblocks one waiting thread (or cv.notify_all())

```


* **Spurious Wakeup Handling:** `cv.wait()` must always pass a boolean predicate (`[&]{ return ready; }`) to prevent execution continuing if woken up by the OS without a valid notification.

---

## III. Summary Checklist for the Oral Exam

1. **Difference between Stack and Heap in Multithreading:** Each thread gets its own isolated stack. The heap is shared among all threads of the same process.
2. **Why use `std::ref` in thread arguments?** `std::thread` constructor passes arguments by value (copy) by default.
3. **What happens if a `std::thread` is destroyed while still joinable?** `std::terminate()` is called, crashing the program.
4. **Why prefer `std::lock_guard` over `std::mutex::lock()`?** To enforce RAII and ensure exception safety (guaranteed unlock on stack unwinding).
5. **How to avoid Deadlocks?** Acquire multiple locks in a strict global sequence across all threads.
6. **Why is a predicate mandatory in `std::condition_variable::wait()`?** To guard against **spurious wakeups** (threads waking without an explicit notification).

---

# WORKSHEET 2: Thread Pool Implementation & Architecture Walkthrough


## 1. Theoretical Motivation & Problem Statement

### The Problem: Thread Creation Overhead

In high-performance numerical computing (such as iterative solvers or Monte Carlo simulations), executing parallel tasks by repeatedly spawning (`std::thread t(...)`) and joining (`t.join()`) native threads introduces severe performance bottlenecks.

* **OS Context Switching & Allocation Costs:** Creating a native C++ thread requires the operating system kernel to allocate dedicated stack memory (typically 1–8 MB) and manage kernel-level thread scheduling structures.


* **High-Frequency Pitfall:** Spawning threads inside loops that execute millions of times causes the thread creation and destruction overhead to dominate actual computational execution time.



### The Solution: The Thread Pool Pattern

A **Thread Pool** resolves this overhead by instantiating a fixed set of $N$ persistent worker threads upon initialization.

* **Task Reuse Model:** Threads are created **once** during program startup and kept alive in a sleeping state.


* **Work Queue:** Tasks are represented as generic callable objects (`std::function<void()>`) and pushed into a shared First-In, First-Out (FIFO) queue (`std::queue`). Worker threads dynamically pull tasks from this queue, execute them, and return to sleep when idle.



---

## 2. Connection to Computer Architecture & Lecture Theory

* **Flynn's Taxonomy:** Thread pools operate under the **MIMD** (Multiple Instruction, Multiple Data) architecture on **Shared Memory Systems**. All worker threads run within the address space of the parent process and share heap memory (the task queue, flags, counters).


* **Private vs. Shared Memory:**
* **Shared (Process Heap):** The task queue (`jobs`), synchronization primitives (`queue_mutex`, condition variables), worker handles vector (`workers`), and state trackers (`active_jobs`, `stop_thread_pool`).


* **Private (Thread Stack):** Local variables created inside worker loops (e.g., the local `job` object retrieved from the queue).




* **Resource Acquisition Is Initialization (RAII):** The lifetime of all worker threads is strictly bound to the lifetime of the `ThreadPool` object. Destructors guarantee graceful pool shutdown and join all threads, preventing dangling execution or system crashes.



---

## 3. Class Components Overview (`thread_pool.hpp`)

| Member Variable | Data Type | Theoretical & Functional Role |
| --- | --- | --- |
| `workers` | `std::vector<std::thread>` | Container holding the persistent worker threads created at pool instantiation.|
| `jobs` | `std::queue<std::function<void()>>` | Thread-safe FIFO task queue storing pending computational jobs.|
| `queue_mutex` | `std::mutex` | Mutual exclusion lock protecting concurrent reads/writes to `jobs`, `active_jobs`, and flags.|
| `queue_condition` | `std::condition_variable` | Signals sleeping worker threads when a new task is enqueued or when shutdown begins.|
| `work_done_condition` | `std::condition_variable` | Signals calling threads blocked in `wait_for_all()` when all enqueued tasks complete.|
| `active_jobs` | `std::size_t` | Atomic/synchronized counter tracking the number of threads currently executing a job.|
| `stop_thread_pool` | `bool` | Boolean shutdown flag instructing worker threads to terminate their event loops.|

---

## 4. Implementation Walkthrough & Code Breakdown

### A. Constructor & Worker Event Loop

The constructor validates thread counts and launches `num_threads` worker execution loops.

```cpp
inline ThreadPool::ThreadPool(const std::size_t num_threads)
{
  if (num_threads == 0)
    throw std::invalid_argument("The number of threads must be greater than zero!");[cite: 6]

  for (std::size_t i = 0; i < num_threads; ++i)
    {
      workers.emplace_back([this]() {
        while (true)
          {
            std::function<void()> job;[cite: 6]

            {
              std::unique_lock<std::mutex> lock(queue_mutex);[cite: 6]

              // Sleep until work is available OR the pool is shutting down
              queue_condition.wait(lock, [this]() {
                return !jobs.empty() || stop_thread_pool;[cite: 6]
              });

              // Shutdown condition: Exit thread loop when pool stops and no jobs remain
              if (stop_thread_pool && jobs.empty())
                {
                  if (active_jobs == 0)
                    work_done_condition.notify_all();[cite: 6]
                  return;[cite: 6]
                }

              // Retrieve task from queue
              job = std::move(jobs.front());[cite: 6]
              jobs.pop();[cite: 6]
              active_jobs++;[cite: 6]
            } // Lock released HERE before job execution![cite: 6]

            // Execute job outside critical section to allow parallel queue access
            job();[cite: 6]

            {
              std::unique_lock<std::mutex> lock(queue_mutex);[cite: 6]
              active_jobs--;[cite: 6]

              // Signal wait_for_all() if all tasks finished and all workers are idle
              if (jobs.empty() && active_jobs == 0)
                work_done_condition.notify_all();[cite: 6]
            }
          }
      });
    }
}

```

#### Key Implementation Details:

1. **Spurious Wakeup Protection:** `queue_condition.wait(lock, predicate)` accepts a lambda predicate (`[!this]() { return !jobs.empty() || stop_thread_pool; }`). This ensures that if the OS wakes up a thread without an explicit notification, the thread re-checks the condition and goes back to sleep.


2. **Critical Section Minimization:** The lock (`queue_mutex`) is released **before** calling `job()`. Executing arbitrary user code inside a critical section would serialize execution, destroying parallelism across worker threads.


3. **Move Semantics:** Standard containers and `std::function` objects are moved (`std::move`) rather than copied to eliminate unnecessary memory allocation overhead.



---

### B. Task Submission (`enqueue`)

Pushes user-defined computational work into the queue and alerts sleeping threads.

```cpp
inline void ThreadPool::enqueue(std::function<void()> job)
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex);[cite: 6]
    jobs.push(std::move(job));[cite: 6]
  }
  queue_condition.notify_one(); // Unblocks a single sleeping worker[cite: 6]
}

```

* **`notify_one()` vs. `notify_all()`:** Because only a single job was added, calling `notify_one()` wakes up exactly one worker thread. Calling `notify_all()` would trigger a **thundering herd problem**, waking up all workers only for all but one to immediately go back to sleep.



---

### C. Synchronization Barrier (`wait_for_all`)

Blocks caller execution until all enqueued tasks are completed and all workers return to an idle state.

```cpp
inline void ThreadPool::wait_for_all()
{
  std::unique_lock<std::mutex> lock(queue_mutex);[cite: 6]

  work_done_condition.wait(lock, [this]() {
    return jobs.empty() && active_jobs == 0;[cite: 6]
  });
}

```

* **Dual Condition Requirement:** Reaching `jobs.empty()` alone is insufficient because jobs currently running in parallel threads are no longer in the queue. The barrier requires **both** `jobs.empty() == true` AND `active_jobs == 0` to confirm full completion.



---

### D. Graceful Shutdown & Destructor

Ensures RAII compliance by terminating worker loops and joining native threads safely.

```cpp
inline ThreadPool::~ThreadPool() noexcept
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex);[cite: 6]
    stop_thread_pool = true;[cite: 6]
  }
  queue_condition.notify_all(); // Wake up ALL sleeping workers so they check stop_thread_pool[cite: 6]

  for (std::thread &worker : workers)
    if (worker.joinable())
      worker.join(); // Block until worker thread exits loop and terminates[cite: 6]
}

```

---

## 5. Summary Table for Oral Exam Defense

| Question / Concept | Defense Explanation |
| --- | --- |
| **Why use a Thread Pool over raw `std::thread`s?** | Eliminates thread creation/destruction OS kernel overhead and context switching costs by creating $N$ persistent workers once and re-using them across tasks.|
| **Why release `queue_mutex` before calling `job()`?** | Holding the mutex during task execution locks the queue, forcing all other worker threads to block sequentially and rendering the system single-threaded.|
| **Why is a predicate necessary in `cv.wait()`?** | Guards against **spurious wakeups** (OS waking a thread without a signal) and race conditions where another thread steals the job first.|
| **Difference between `queue_condition` and `work_done_condition`?** | `queue_condition` signals **workers** that tasks are available; `work_done_condition` signals **external callers** (`wait_for_all`) that all tasks are finished.|
| **What happens if a thread pool is destroyed while tasks are active?** | The destructor sets `stop_thread_pool = true`, wakes sleeping threads, allows running tasks to complete, and joins all workers gracefully (RAII).|


---

# Open MP

### 1. Architecture & Execution Model

* **Parallel Framework**: OpenMP (Open Multi-Processing) is a standardized API for shared-memory parallel programming in C, C++, and Fortran. It targets MIMD (Multiple Instruction, Multiple Data) architectures.


* **Three Key Components**:
1. **Compiler Directives**: Pragma directives (`#pragma omp ...`) that instruct the compiler to parallelize code blocks.


2. **Runtime Library Routines**: Functions accessed via `#include <omp.h>` (e.g., `omp_get_thread_num()`, `omp_get_num_threads()`).


3. **Environment Variables**: External configurations affecting execution runtime behavior (e.g., `OMP_NUM_THREADS`).




* **Compilation Flag**: Requires enabling compiler flag `-fopenmp` during compilation.


* **Fork-Join Execution Model**:
* Execution begins with a single **master thread**.


* Upon reaching a parallel construct, the master thread **forks** a team of threads.


* At the end of the construct, threads **join** back together, and only the master thread continues sequentially.





---

### 2. Loop Scheduling Strategies

OpenMP loop iterations are distributed among threads using different scheduling policies specified via the `schedule()` clause:

| Scheduling Policy | Syntax Example | Key Use Case & Behavior |
| --- | --- | --- |
| **Static** | `schedule(static, chunk_size)`<br> | Best when iteration execution time is constant and predictable. Chunks are assigned round-robin at compile-time/start.|
| **Dynamic** | `schedule(dynamic, chunk_size)`<br> | Best when iteration execution time varies significantly. Chunks are assigned to idle threads dynamically at runtime.|
| **Guided** | `schedule(guided, min_size)`<br> | Hybrid approach where chunk sizes start large and exponentially decrease down to `min_size` as work completes.|

---

## Data-Sharing Attributes & Scoping Rules

When entering a parallel region, data scope must be carefully managed to avoid race conditions or unexpected state modifications:

* **`shared(var)`**: A single memory location is accessed and modified by all threads in the team.


* **`private(var)`**: Each thread gets its own uninitialized copy of the variable. The original value outside the region is unchanged.


* **`firstprivate(var)`**: Each thread gets its own private copy initialized with the value the variable had before entering the parallel region.


* **`lastprivate(var)`**: The private value from the logically last loop iteration is written back to the outer shared variable after the loop finishes.


* **`default(none)`**: Forces the programmer to explicitly state the scope of every variable used inside the parallel region, preventing accidental scoping bugs.


* **`reduction(op : var)`**: Creates local private copies of `var` per thread, performs operations locally, and combines them into the global `var` using operator `op` (e.g., `+`, `min`, `max`) at loop completion.



---

## Code Syntax Breakdown (Live Coding Examples)

### 1. Parallel Region & Nesting

```cpp
omp_set_max_active_levels(2); // Enables nested parallel regions up to 2 levels deep[cite: 1]

#pragma omp parallel num_threads(2) // Spawns a parallel region with 2 threads[cite: 1]
{
    int id = omp_get_thread_num(); // Returns current thread ID[cite: 1]
    int total = omp_get_num_threads(); // Returns total thread count[cite: 1]
    
    #pragma omp parallel num_threads(2) // Nested parallel region[cite: 1]
    {
        // Executes inside child threads[cite: 1]
    }

    #pragma omp barrier // Explicit barrier: all threads wait here before proceeding[cite: 1]
}

```

### 2. Synchronization & Single-Threaded Sections

* **Mutual Exclusion (`critical`)**:
```cpp
#pragma omp critical // Ensures only 1 thread executes this block at a time[cite: 1]
{
    ++counter; // Thread-safe state update[cite: 1]
}

```


* **Single Execution (`single`)**:
```cpp
#pragma omp single // Executed by only ONE thread; other threads wait at implicit barrier[cite: 1]
{
    std::cout << "Initialization task by one thread\n";[cite: 1]
} // Implicit barrier unless 'nowait' is added[cite: 1]

```



### 3. Loop Worksharing & Advanced Loop Directives

* **Automatic Worksharing Loop (`omp for`)**:
```cpp
#pragma omp for schedule(static, 73)[cite: 1]
for (std::size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];[cite: 1]
}

```


* **Loop Reduction & `nowait`**:
```cpp
// 'nowait' removes the implicit barrier at the end of the loop construct[cite: 1]
#pragma omp for reduction(max : max_val) schedule(dynamic, 10) nowait[cite: 1]
for (std::size_t i = 0; i < n; ++i) {
    max_val = std::max(max_val, c[i]);[cite: 1]
}

```


* **Loop Collapsing (`collapse(N)`)**:
```cpp
// Combines 2 nested loops into 1 single multi-dimensional iteration space[cite: 1]
#pragma omp parallel for num_threads(2) schedule(static, 1) collapse(2)[cite: 1]
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
        // Process element (i, j)[cite: 1]
    }
}

```


* **Enforced Iteration Order (`ordered`)**:
```cpp
#pragma omp parallel for ordered num_threads(4)[cite: 1]
for (int i = 0; i < 8; ++i) {
    // Parallel computations[cite: 1]

    #pragma omp ordered // Forces execution sequentially in logical order of i[cite: 1]
    {
        std::cout << "Processing item " << i << "\n";[cite: 1]
    }
}

```

---

## Key Oral Exam & Worksheet Takeaways

> ### Essential Exam Takeaways
> 
> 
> 1. **Default Synchronization Points**: Most worksharing directives (`#pragma omp for`, `#pragma omp single`) feature an **implicit barrier** at the end unless explicitly overridden using the `nowait` clause.
> 
> 
> 2. **Race Conditions vs. Atomic/Critical**: Shared updates without `#pragma omp critical`, `#pragma omp atomic`, or `reduction` lead to data races.
> 
> 
> 3. **Data Scoping Rigor**: Always use `default(none)` in complex regions to prevent uninitialized private variables (`private`) or unexpected modifications to shared variables (`shared`).
> 
> 
> 4. **Scheduling Tradeoffs**: `static` scheduling has minimal overhead but suffers from load imbalance if iterations take varying time. `dynamic` fixes load imbalance but introduces runtime overhead for chunk distribution.
> 
> 
> 
>

---

# WORKSHEET 2 BREAKDOWN

## 1. Task 1: Linear Algebra Vector Parallelization (`LaVector<number>`)

### Implementation Pattern: OpenMP Worksharing & Scoping

In `la_vector.cpp`, vector operators (`operator+`, `operator+=`, `operator-`, `operator-=`, `operator*`, `operator*=`, `operator/`, `operator/=`, `schur_product`) and vector norms are parallelized using the `#pragma omp parallel for` directive.

To adhere to OpenMP best practices and eliminate subtle scoping bugs, every parallel construct explicitly enforces `default(none)` and explicitly declares all variables as either `shared` or via `reduction`:

```cpp
// Example: Element-wise addition (operator+)
#pragma omp parallel for default(none) shared(self, vec, result, n)
for (std::size_t i = 0; i < n; ++i)
    result[i] = self[i] + vec[i];

```

---

### Reductions & Norm Computations

For scalar outputs derived from vector traversals (`dot_product`, `l2_norm_squared`, `norm_l1`, and `norm_linf`), OpenMP reduction clauses are utilized to prevent data races without using expensive critical sections or explicit locks:

* **Dot Product & $L_2$ Norm Squared**:
```cpp
#pragma omp parallel for default(none) shared(self, other, n) reduction(+ : result)
for (std::size_t i = 0; i < n; ++i)
    result += self[i] * other[i];

```


* **$L_1$ Norm**:
```cpp
#pragma omp parallel for default(none) shared(self, n) reduction(+ : sum)
for (std::size_t i = 0; i < n; ++i)
    sum += std::abs(self[i]);

```


* **$L_\infty$ Norm**:
```cpp
#pragma omp parallel for default(none) shared(self, n) reduction(max : max_abs)
for (std::size_t i = 0; i < n; ++i) {
    const number a = std::abs(self[i]);
    if (a > max_abs) max_abs = a;
}

```



---

### Oral Exam Defense Questions (Task 1)

> #### Q1: Why is `schedule(static)` implicitly optimal for `LaVector` operations?
> 
> 
> * **Reason**: Each loop iteration performs an identical $O(1)$ arithmetic operation on contiguous memory addresses. Workload distribution is perfectly uniform across threads, making dynamic runtime scheduling unnecessary and wasteful due to queue access overhead.
> 
> 
> 
> 

> #### Q2: Why does increasing the thread count fail to yield linear speedup for large vector operations?
> 
> 
> * **Memory Wall / Bandwidth Bottleneck**: Simple vector operations (e.g., $c_i = a_i + b_i$) have a very low **operational intensity** (ratio of FLOPs to bytes accessed). The CPU execution cores process data faster than main memory (DRAM) can supply it, causing thread execution time to be bounded by memory bus bandwidth rather than raw compute power.
> 
> 
> 
> 

---

## 2. Task 2: Linear Algebra Matrix Parallelization (`LaMatrix<number>`)

### A. Element-Wise Operations & Loop Collapsing (`collapse(2)`)

For 2D array structures stored as `std::vector<std::vector<number>>`, element-wise additions, subtractions, and scalar operations iterate across both rows and columns. The `collapse(2)` clause merges the two nested loops into a single multi-dimensional iteration space:

```cpp
#pragma omp parallel for default(none) shared(self, mat, result, num_rows, num_cols) collapse(2)
for (std::size_t row_idx = 0; row_idx < num_rows; ++row_idx)
    for (std::size_t col_idx = 0; col_idx < num_cols; ++col_idx)
        result[row_idx, col_idx] = self[row_idx, col_idx] + mat[row_idx, col_idx];

```

#### Why use `collapse(2)` here?

* **Larger Iteration Space**: If a matrix has a small row count (e.g., $4 \times 10000$), parallelizing only the outer loop leaves threads underutilized. Collapsing combines $M \times N$ iterations into a single flat work pool, drastically improving load distribution across threads.



---

### B. Matrix-Matrix Multiplication ($C = A \cdot B$)

Matrix-matrix multiplication uses an optimized $i \to k \to j$ loop ordering (`row_idx` $\to$ `idx` $\to$ `col_idx`) to ensure linear, cache-friendly memory accesses along the contiguous rows of matrix $B$:

```cpp
#pragma omp parallel for default(none) shared(self, mat, result, num_rows, num_cols, mat_num_cols)
for (std::size_t row_idx = 0; row_idx < num_rows; ++row_idx)
    for (std::size_t idx = 0; idx < num_cols; ++idx)
        for (std::size_t col_idx = 0; col_idx < mat_num_cols; ++col_idx)
            result[row_idx, col_idx] += self[row_idx, idx] * mat[idx, col_idx];

```

#### Why parallelize ONLY the outermost loop ($i$)?

1. **Thread Safety**: Each thread handles a distinct subset of rows in matrix $C$ (`row_idx`). Since write targets `result[row_idx, col_idx]` are unique per outer iteration, no data races occur on matrix $C$, eliminating the need for `critical` or `atomic` constructs.


2. **Minimal Thread Management Overhead**: Forking and synchronizing threads once at the outer loop level avoids repeating parallel region overhead $M$ times.



---

### C. Matrix Trace & Matrix-Vector Product

* **`trace()`**: Operates purely along the main diagonal ($A_{i, i}$) using a single loop parallelized via `reduction(+ : tr)`.


* **Matrix-Vector Product ($y = A \cdot x$)**: Parallelized over matrix rows (`row_idx`). Each thread computes the complete inner product for row `row_idx` into `result[row_idx]`, guaranteeing thread-safe writes.



---

### Oral Exam Defense Questions (Task 2)

> #### Q1: Why shouldn't you apply `collapse(3)` to matrix-matrix multiplication?
> 
> 
> * **Reason**: Collapsing the inner $k$ or $j$ loops causes multiple threads to concurrently attempt updates to the same output cell $C_{i, j}$, introducing severe data races that require explicit mutex locking (`critical`) or atomic operations, drastically ruining performance.
> 
> 
> 
> 

> #### Q2: What is the difference between CPU Time and Wall-Clock Time in parallel benchmarks?
> 
> 
> * **Wall-Clock Time**: Real elapsed time from start to end of execution.
> 
> 
> * **CPU Time**: Total active execution time accumulated across all CPU cores. In an ideal $N$-thread parallel region running for 1 second of wall-clock time, CPU time equals $N$ seconds.
> 
> 
> 
> 

---

## 3. Task 3: Matrix-Free Heat Equation Solver (`HeatOperator`)

In the matrix-free explicit Euler implementation, the state update $u_i^{(n+1)} = u_i^{(n)} + \Delta t \cdot \left(\text{stencil}(i, u^{(n)}) + Q_i\right)$ is computed directly on degrees of freedom (DoFs) without forming or storing a global matrix:

```cpp
#pragma omp parallel for default(none) shared(heat_data, dof_handler, current_time_increment, old_time, old_solution, current_solution)
for (std::size_t i = 0; i < heat_data.grid.n_global_dofs; ++i) {
    if (!dof_handler.at_boundary(i)) {
        current_solution[i] = old_solution[i] + current_time_increment * 
            (matrix_free_stencil(i, old_solution) + source_term(old_time, i));
    }
}

```

Similarly, the source term evaluation `source_term_vector` is parallelized across all global DoF indices:

```cpp
#pragma omp parallel for default(none) shared(heat_data, current_time, source)
for (std::size_t i = 0; i < heat_data.grid.n_global_dofs; ++i)
    source[i] = source_term(current_time, i);

```

---

### Oral Exam Defense Questions (Task 3)

> #### Q1: Why can a matrix-free solver outperform a matrix-based solver even though it recomputes finite difference stencils on every time step?
> 
> 
> * **Reason**: Matrix-based updates require fetching giant sparse matrix structures from main memory on every time step, starving CPU execution units due to limited memory bandwidth. The matrix-free approach trades arithmetic operations (which modern CPUs execute rapidly in registers) for memory transfers, significantly reducing memory bus traffic and overcoming the Memory Wall bottleneck.
> 
> 
> 
> 

---

## 4. Task 4: System Matrix Initialization (`setup_finite_difference_heat_equation_system_matrix`)

The assembly loop constructs the discrete Laplace operator matrix row-by-row:

```cpp
#pragma omp parallel for default(none) shared(heat_data, dof_handler, system_matrix, offset) schedule(static)
for (std::size_t row_idx = 0; row_idx < heat_data.grid.n_global_dofs; ++row_idx) {
    if (!dof_handler.at_boundary(row_idx)) {
        // Compute stencil coefficients and assign entries in system_matrix[row_idx, ...]
    }
}

```

---

### Oral Exam Defense Questions (Task 4)

> #### Q1: Is explicit synchronization (`#pragma omp critical`) required during system matrix assembly?
> 
> 
> * **No**: Each loop iteration `row_idx` writes exclusively to row `row_idx` of `system_matrix`. Because `LaMatrix` stores rows as independent allocations (`std::vector<std::vector<number>>`), threads access disjoint memory locations without write conflicts.
> 
> 
> 
> 

> #### Q2: Which scheduling policy is best suited for system matrix assembly?
> 
> 
> * **`schedule(static)`**: Assembly workload per interior DoF is fixed and identical. Static partitioning eliminates dynamic queue overhead and achieves optimal workload distribution.
> 
> 
> 
> 

---

## 5. Task 5: Implicit Time Stepping with the Jacobi Method

### Mathematical Formulation

The implicit Euler time-stepping scheme requires solving the linear system $A x = b$ at each step:


$$A := I - \Delta t K, \quad x := u^{n+1}, \quad b := u^n + q^{n+1}$$

The Jacobi iterative method updates the solution component-wise via:


$$x_i^{(k+1)} = x_i^{(k)} + \frac{b_i - (A x^{(k)})_i}{A_{ii}}$$

where the diagonal entry $A_{ii}$ is given by:


$$A_{ii} = 1 - \Delta t K_{ii} = 1 + \Delta t \sum_{d=1}^{\text{dim}} \frac{2 \kappa_i}{\Delta x_d^2}$$

---

### Matrix-Free Jacobi Algorithm Structure

```cpp
// 1. Parallel RHS setup: b = u^n + q^(n+1)
#pragma omp parallel for default(none) shared(heat_data, dof_handler, rhs, current_time)
for (std::size_t i = 0; i < heat_data.grid.n_global_dofs; ++i)
    if (!dof_handler.at_boundary(i))
        rhs[i] += source_term(current_time, i);

// 2. Iterative Jacobi Loop (max 1000 iterations)
for (std::size_t iter = 0; iter < 1000; ++iter) {
    // 2a. Parallel Residual Calculation: r_i = (A * x_old)_i - b_i
    #pragma omp parallel for default(none) shared(heat_data, dof_handler, current_time_increment, x_old, rhs, residual_vec)
    for (std::size_t i = 0; i < heat_data.grid.n_global_dofs; ++i) {
        if (!dof_handler.at_boundary(i)) {
            number A_x_i = x_old[i] - current_time_increment * matrix_free_stencil(i, x_old);
            residual_vec[i] = A_x_i - rhs[i];
        } else {
            residual_vec[i] = 0.0;
        }
    }

    // 2b. Convergence Check: relative residual norm ||r||_2 / ||x_old||_2
    number r = residual_vec.norm_l2() / x_old.norm_l2();
    if (iter > 0 && r < heat_data.iterative_solver_tolerance)
        break;

    // 2c. Parallel Jacobi Point Update
    #pragma omp parallel for default(none) shared(heat_data, dof_handler, current_time, current_time_increment, x_old, x_new, rhs)
    for (std::size_t i = 0; i < heat_data.grid.n_global_dofs; ++i) {
        if (!dof_handler.at_boundary(i)) {
            x_new[i] = jacobi_update(i, current_time, current_time_increment, x_old, rhs);
        }
    }

    // 2d. Double Buffering Swap
    x_old = x_new;
}

```

---

### Oral Exam Defense Questions (Task 5)

> #### Q1: Why is double buffering (`x_old` and `x_new`) mandatory in the parallel Jacobi method?
> 
> 
> * **Reason**: The Jacobi method requires that all updates in iteration $(k+1)$ depend strictly on values from iteration $(k)$. If updates were written in-place to `x_old`, threads would read partially updated values from neighboring indices, converting the algorithm into a non-deterministic parallel Gauss-Seidel scheme and causing race conditions.
> 
> 
> 
> 

---

## 6. Summary Table for Oral Exam Defense

| Operation / Construct | Implementation Pattern | Key Reason / Strategy |
| --- | --- | --- |
| **`LaVector` Norms & Dot** | `#pragma omp parallel for reduction(...)`<br> | Prevents data races on global scalar accumulators.|
| **Matrix Element-Wise** | `#pragma omp parallel for collapse(2)`<br> | Merges row/col loops for maximum thread work distribution.|
| **Matrix Multiplication** | Outer loop parallelized ($i$), order $i \to k \to j$<br> | Ensures contiguous memory access on $B$ without output races.|
| **Matrix-Free Solver** | Recomputes stencil dynamically in parallel loop| Mitigates memory bandwidth limits (Memory Wall).|
| **Matrix Setup** | Parallel over rows, `schedule(static)`<br> | Thread-safe independent row writes; equal workload per DoF.|
| **Jacobi Solver** | Separate parallel steps + double buffering| Guarantees algorithmic correctness without inter-thread dependencies.|


---

# MPI - Message Passing Interface

### 1. Distributed Memory Architecture & Process Model

* **Distributed Memory Systems**: Each process runs on its own node/processor and possesses its own private memory address space. Processes cannot directly read or write to another process's memory; data must be exchanged explicitly across a network interconnect (e.g., InfiniBand).


* **MPI vs. OpenMP**: OpenMP uses a shared-memory model with a single process spawning multiple threads sharing the same address space. MPI launches $N$ separate, fully isolated processes across nodes, each running its own copy of the executable.


* **Environment & Communicators**:
* An MPI environment must be initialized before any MPI function calls and finalized upon exit.


* **Communicator**: A group of processes that can communicate with each other. `MPI_COMM_WORLD` is the default communicator encompassing all launched processes.


* **Rank**: The unique integer identifier assigned to each process inside a communicator (from $0$ to $\text{size}-1$).





---

### 2. Point-to-Point Communication & Underlying Protocols

Point-to-point transfers happen between a specific sender rank and receiver rank using matched parameters (Communicator, Tag, Source/Destination).

#### Under-the-Hood Transfer Protocols (Critical Oral Exam Topic)

When `MPI_Send` is invoked, MPI dynamically chooses between two primary protocols based on message size:

* **Eager Protocol (Small / Medium Messages)**:


* Data is sent immediately to the receiver's MPI System Buffer without waiting for the receiver to post a matching `MPI_Recv`.


* **Pros**: Sender returns quickly, minimizing CPU waiting time.


* **Cons**: Requires extra memory overhead and buffer copies on the receiving system.




* **Rendezvous Protocol (Large Messages)**:


* A light-weight handshake takes place first: sender asks receiver if it is ready. Once receiver posts a matching `MPI_Recv`, it acknowledges the sender, and data is transferred **directly** from sender memory to receiver program memory.


* **Pros**: Eliminates double-buffering and system buffer memory overhead.


* **Cons**: Requires additional handshake round-trips; sender blocks until the complete transfer finishes.





#### Message Queues Inside MPI

* **Receive Queue**: Holds requests posted by `MPI_Recv` that have not yet matched incoming data.


* **Unexpected Message Queue (UMQ)**: Stores incoming data packets (via Eager protocol) that arrived before a matching `MPI_Recv` was called.



#### Deadlocks

Deadlocks occur when processes wait indefinitely for each other to complete synchronous communication calls. For example, if Rank 0 and Rank 1 both call blocking `MPI_Recv` first, neither can send data.

---

### 3. Non-Blocking Communication

* **Purpose**: Allows communication to happen in the background while the CPU performs independent computation, enabling **computation-communication overlap** and avoiding deadlocks.


* **Request Handles (`MPI_Request`)**: Non-blocking functions (`MPI_Isend`, `MPI_Irecv`) return immediately and use an `MPI_Request` handle to track progress.


* **Synchronization / Completion**:
* `MPI_Wait`: Blocks execution until the non-blocking operation associated with the request completes.


* `MPI_Test`: Non-blocking query that polls whether the operation is finished (returns a boolean flag).





---

### 4. MPI Collective Operations

Collective operations involve **all processes** in a communicator. If one process omits a collective call, the program will hang indefinitely.

| Collective Function | Operation Description |
| --- | --- |
| `MPI_Barrier` | Blocks processes until all processes in the communicator reach this point.|
| `MPI_Bcast` | Broadcasts data from a single root rank to all other ranks.|
| `MPI_Reduce` | Combines data from all ranks using an operation (e.g., `MPI_SUM`, `MPI_MIN`, `MPI_MAX`) and stores the result on the root rank.|
| `MPI_Allreduce` | Performs reduction and distributes the final result to **all** ranks.|
| `MPI_Gather` / `MPI_Gatherv` | Collects distinct data blocks from all ranks and concatenates them on the root rank.|
| `MPI_Scatter` / `MPI_Scatterv` | Splits a data buffer on the root rank into equal (or variable) chunks and distributes them across ranks.|
| `MPI_Alltoall` | Performs an all-to-all personalized exchange where every rank sends distinct data to every rank.|

---

## MPI Live Coding Syntax Reference

Below is a reference of the core C++ MPI syntax derived from the live coding sessions:

### Environment Setup & Basic Queries

```cpp
#include <mpi.h> // Required header[cite: 1]

int main(int argc, char** argv) {
    // 1. Initialize MPI environment[cite: 1]
    MPI_Init(&argc, &argv);

    int rank, size;
    // 2. Get current process ID (rank)[cite: 1]
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 3. Get total process count[cite: 1]
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 4. Abort MPI execution on error[cite: 1]
    if (size < 2) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 5. Finalize MPI environment (no MPI calls allowed after this)[cite: 1]
    MPI_Finalize();
    return 0;
}

```

### Point-to-Point Syntax

#### Blocking Point-to-Point

```cpp
int data = 100;
int tag = 0;

// Sender (e.g., Rank 0)
MPI_Send(&data, 1, MPI_INT, 1 /* dest */, tag, MPI_COMM_WORLD);[cite: 1]

// Receiver (e.g., Rank 1)
MPI_Recv(&data, 1, MPI_INT, 0 /* src */, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);[cite: 1]

```

#### Non-Blocking Point-to-Point & Completion

```cpp
MPI_Request request;
MPI_Status status;
int data = 747;

// Non-blocking Send[cite: 1]
MPI_Isend(&data, 1, MPI_INT, 1 /* dest */, tag, MPI_COMM_WORLD, &request);

// Non-blocking Receive[cite: 1]
MPI_Irecv(&data, 1, MPI_INT, 0 /* src */, tag, MPI_COMM_WORLD, &request);

// Polling for completion (Non-blocking)[cite: 1]
int flag = 0;
MPI_Test(&request, &flag, &status);[cite: 1]
if (flag) {
    // Communication complete[cite: 1]
}

// Explicit Wait (Blocking until complete)[cite: 1]
MPI_Wait(&request, &status);[cite: 1]

```

---

## Key Takeaways for Oral Exam & Worksheets

### Oral Exam Preparation Points

1. **Explain Eager vs. Rendezvous Protocol**: Be prepared to explain how buffer sizes dictate protocol choice, what memory trade-offs occur, and why Rendezvous avoids memory overhead at the expense of handshake latency.


2. **Explain the UMQ vs. Receive Queue**: Explain what happens if `MPI_Send` arrives before `MPI_Recv` is called (stored in Unexpected Message Queue) versus when `MPI_Recv` is posted first (stored in Receive Queue).


3. **Deadlock Identification**: Identify deadlocks in code examples involving synchronous blocking communications.


4. **Shared vs. Distributed Memory**: Contrast threads (shared heap, separate stacks) with processes (isolated address space, explicit message passing).


5. **Collective Synchronization Requirements**: Explain why all processes must call a collective operation and the performance implications of barrier synchronization.



### Key Concepts for Worksheets & Domain Decomposition

* **Ghost Cells / Halo Exchange**: In parallel finite difference or grid simulations (such as the heat equation), each rank owns a portion of the domain. Boundaries between ranks require exchanging boundary values ("ghost cells") via point-to-point communication.


* **Hiding Communication Latency**:
1. Post non-blocking receives (`MPI_Irecv`) and sends (`MPI_Isend`) for ghost layer data.


2. Compute updates for the **inner domain** (data that doesn't depend on ghost cells) while communication proceeds in the background.


3. Call `MPI_Wait` / `MPI_Waitall` to ensure ghost cells have arrived.


4. Compute updates for the **outer domain boundaries** using the freshly received ghost values.




* **Buffer Safety**: Never overwrite or re-use a memory buffer passed to `MPI_Isend` or `MPI_Irecv` until `MPI_Wait` or `MPI_Test` confirms the request is complete.





# WORKSHEET 3

Here is a comprehensive, step-by-step walkthrough of **Worksheet 3 (MPI Parallel Heat Solver)** based on your actual source code and submitted report.

This walkthrough is specifically tailored for your **oral exam**, highlighting the architectural decisions, theoretical reasoning, MPI mechanisms, and potential exam questions.

---

# Strategic Overview

### Core Objective

Transform a serial, 1D/2D/3D matrix-free Heat Equation solver into a fully distributed parallel solver using **MPI domain decomposition** (slicing the spatial domain along the highest spatial dimension).

### Key Abstractions

1. **`Partitioner`**: The communication backbone. Determines communication topology (who needs data from whom) and manages asynchronous ghost cell exchanges.
2. **`ParallelDistributedVector`**: Linear algebra vector that holds `local_elements` (locally owned elements followed by read-only ghost elements).
3. **`DoFHandler`**: Maps physical/cartesian grid indices to global 1D array indices and handles domain splitting (including process oversubscription).
4. **`HeatOperator`**: Implements explicit and implicit (Jacobi) time-stepping schemes using matrix-free stencils.

---

# Task-by-Task Walkthrough & Solution Analysis

## Task 1: Distributed Vector & Linear Algebra Operations

**File:** `parallel_distributed_vector.cpp`

### 1. Vector Memory Layout

* Each process holds a continuous array `local_elements` containing:

$$\text{local\_elements} = [\underbrace{\text{Locally Owned Entries}}_{0 \dots N_{\text{owned}}-1} \,\vert{}\, \underbrace{\text{Ghost Entries}}_{N_{\text{owned}} \dots N_{\text{owned}}+N_{\text{ghost}}-1}]$$



### 2. Local vs. Global Vector Operations

* **Local Operations (`+=`, `-=`, `*=`, `+`, `-`)**:
* Carried out entirely locally across all `local_elements` (owned + ghost).
* **No MPI communication is needed.**
* *Prerequisite check*: `assert_compatibility()` verifies that both vectors share the exact same process distribution and ghost layout.


* **Global Operations (`dot_product`, `l2_norm`)**:
* **Critical Detail**: Sum **ONLY** the locally owned entries ($0 \dots N_{\text{owned}}-1$).
* *Why?* Ghost elements are copies of data owned by neighboring processes. Summing ghost elements would **double-count** values, producing an incorrect dot product.
* *MPI Call*: `MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, communicator);`
* `l2_norm()` simply computes $\sqrt{\text{dot\_product}(*\text{this})}$.



---

## Task 2: The MPI Partitioner & Ghost Exchange

**Files:** `partitioner.cpp`, `partitioner.hpp`

The partitioner solves the problem of establishing **which rank needs to send/receive which indices to/from whom**, without central master coordination.

### 1. Determining Communication Pattern (`determine_communication_pattern`)

```
   Rank 0 Requests Ghosts          Rank 1 Requests Ghosts          Rank 2 Requests Ghosts
             │                               │                               │
             └───────────────────────┬───────┴───────────────────────────────┘
                                     ▼
                      Step 1: MPI_Allgather & MPI_Allgatherv
                                     │
                                     ▼
                      Global Array of All Requested Ghosts
                                     │
                                     ▼
                   Step 2: Local Ownership Check (Filtering)
                                     │
                                     ▼
                  Fills export_targets_and_indices (Who I Send To)
                                     │
                                     ▼
                   Step 3: Transpose via MPI_Alltoall
                                     │
                                     ▼
                 Fills import_targets_and_indices (Who I Recv From)

```

* **Step 1: Ghost Request Phase (Global Gathering)**
1. Each rank queries its local ghost count: `my_ghost_count = ghost_indices.size()`.
2. `MPI_Allgather` distributes the ghost counts to all processes (`ghost_counts`).
3. Displacements (`ghost_displs`) are computed.
4. `MPI_Allgatherv` gathers all requested ghost indices from all processes into a global flattened array `all_ghosts`.


* **Step 2: Ownership Check Phase**
1. Each rank loops through the collected ghost requests of every other rank $r$.
2. If the current rank owns an index requested by rank $r$ (`locally_owned_indices.is_element(index)`), it adds that index to a temporary set.
3. Populates `export_targets_and_indices`: pairs of `(rank_r, index_set)`.


* **Step 3: Communication Matrix Transpose & Asynchronous Setup**
* *Problem*: Rank A knows what it needs to *export* to Rank B, but Rank B does not yet know what it needs to *import* from Rank A.
* *Solution*:
1. Build `send_sizes` vector where `send_sizes[r]` is the number of indices to send to rank $r$.
2. Execute `MPI_Alltoall(send_sizes, 1, ..., recv_sizes, 1, ...)` to transpose send counts into expected receive counts (`recv_sizes`).
3. Allocate receive buffers and post **non-blocking receives** (`MPI_Irecv`).
4. Post **non-blocking sends** (`MPI_Isend`) for outgoing data.
5. Call `MPI_Waitall` to synchronize, then wrap raw received buffers into `import_targets_and_indices`.





### 2. Exchanging Ghost Values (`export_to_ghosted_array_start` / `finish`)

* Called during solver time-stepping to update ghost boundaries.
* **Non-blocking strategy**:
* Posts `MPI_Irecv` directly into the memory location of `ghost_array[local_ghost_idx]`.
* Posts `MPI_Isend` directly from `locally_owned_array[local_owned_idx]`.


* **Tag Disambiguation Trick**: The **global index** of the element is passed as the `tag` parameter in `MPI_Isend`/`MPI_Irecv`. This guarantees that arriving messages match the exact memory location regardless of network arrival order.
* `export_to_ghosted_array_finish`: Calls `MPI_Waitall(request.size(), request.data(), ...)` to complete transfers.

---

## Task 3: Oversubscription & Sub-Communicator Splitting

**File:** `dof_handler.cpp` (`initialize_dof_vector`)

### The Oversubscription Problem

If a user launches a simulation with $P$ processes, but the domain along the split direction (the last dimension $z$ in 3D or $y$ in 2D) has only $N_{\text{slices}} < P$ grid points, some processes will have 0 elements assigned. In standard partitioning, this leads to invalid index ranges and crashes.

### The Solution: `MPI_Comm_split`

1. Compute `max_slices = heat_data.grid.n_dofs[dim - 1]`.
2. If `size > max_slices`:
* Active ranks (`rank < max_slices`) set `color = 0`.
* Idle ranks (`rank >= max_slices`) set `color = MPI_UNDEFINED`.


3. Call `MPI_Comm_split(comm, color, rank, &new_comm)`.
4. Ranks passing `MPI_UNDEFINED` receive `new_comm = MPI_COMM_NULL`.
* Idle ranks construct an empty `ParallelDistributedVector(MPI_COMM_NULL)` and return early.


5. Active ranks update `comm = new_comm`, re-query `comm_rank` and `comm_size`, and partition the grid normally.

---

## Task 4: Heat Operator Parallelization

**File:** `heat_operator.cpp`

### 1. Matrix-Free Stencil Computation (`matrix_free_stencil`)

* Calculates discrete spatial derivatives without constructing global sparse matrices.
* Lookups for center and neighbor values use `solution.global_element(global_index ± offset)`.
* **`global_element()` Lookup Logic**:
* If the global index is locally owned $\rightarrow$ returns direct reference from local memory.
* If the global index is a ghost $\rightarrow$ resolves local offset in the ghost buffer.



### 2. Explicit Euler Scheme (`advance_time_step`)

```cpp
current_solution.swap(old_solution);
old_solution.update_ghost_values(); // Exchange halo cells across MPI ranks

for (unsigned local_index = 0; local_index < current_solution.locally_owned_size(); ++local_index) {
    unsigned global_index = partitioner.local_to_global(local_index);
    if (!dof_handler.at_boundary(global_index)) {
        current_solution.local_element(local_index) = old_solution.local_element(local_index) 
            + dt * (matrix_free_stencil(global_index, old_solution) + source_term(...));
    }
}

```

### 3. Implicit Euler Scheme with Jacobi Iteration (`advance_time_step`)

* Requires iterative updates until global residual norm $\le \text{tol}$.
* **Inside the Jacobi Loop**:
1. Synchronize iteration ghosts: `old_iter.update_ghost_values()`.
2. Compute local Jacobi update: `jacobi_update(...)`.
3. Synchronize new solution ghosts: `current_solution.update_ghost_values()`.
4. Compute local sum of squared residuals (`loc_res * loc_res`).
5. **Global Residual Reduction**:
```cpp
MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, comm);

```


* `MPI_IN_PLACE` is required because `&res` acts as both input buffer and output buffer on the calling rank.





---

# Oral Exam Preparation: Key Questions & Answers

### Q1: Why can’t we sum all entries (including ghosts) during `dot_product`?

> **Answer:** Ghost entries are read-only copies of elements owned by neighboring processes. If every process summed its entire array (owned + ghost), shared boundary values would be counted twice (or multiple times in higher dimensions). To maintain mathematical correctness, each process must sum **only** its `locally_owned_size()` elements before performing `MPI_Allreduce`.

### Q2: Walk me through `determine_communication_pattern()`. How do ranks know who to receive ghost values from?

> **Answer:**
> 1. First, ranks exchange their ghost index requests globally using `MPI_Allgather` (for request counts) and `MPI_Allgatherv` (for flattened requested indices).
> 2. Each rank checks all requested indices against its `locally_owned_indices` to build its `export_targets_and_indices` (who it must send data to).
> 3. To find out who it will *receive* data from, each rank creates a `send_sizes` vector and transposes it across all ranks using `MPI_Alltoall`. The resulting `recv_sizes` vector tells each process exactly how many elements to expect from every other rank, allowing it to allocate receive buffers and post non-blocking `MPI_Irecv` calls.
> 
> 

### Q3: Why did you use the global index as the message `tag` in non-blocking ghost updates?

> **Answer:** Non-blocking communications (`MPI_Isend` / `MPI_Irecv`) can complete out of order depending on network routing. By setting the message tag equal to the element's unique `global_index`, MPI ensures that incoming messages match the exact destination memory address in `ghost_array`, preventing race conditions and buffer corruption.

### Q4: What happens if a user runs a 3D simulation with 64 MPI processes, but there are only 16 grid slices along the $z$-axis?

> **Answer:** The code handles process oversubscription gracefully in `dof_handler.cpp` using `MPI_Comm_split`. It compares process size against `max_slices` ($N_{z}$). The first 16 ranks pass `color = 0` to form an active sub-communicator, while ranks $\ge 16$ pass `color = MPI_UNDEFINED`. Ranks passing `MPI_UNDEFINED` receive `MPI_COMM_NULL`, initialize an empty vector, and sit idle safely without crashing the execution.

### Q5: What is `MPI_IN_PLACE` in `MPI_Allreduce` and why did you use it?

> **Answer:** Standard `MPI_Allreduce` expects separate send and receive memory buffers (`sendbuf` and `recvbuf`). When evaluating the implicit solver's residual, we accumulate the local squared residual directly into a scalar variable `res`. By passing `MPI_IN_PLACE` as the send buffer and `&res` as the receive buffer, we comply with the C/MPI standard and avoid memory overlap undefined behavior.

### Q6: How does non-blocking ghost exchange allow computation-communication overlap?

> **Answer:** By calling `export_to_ghosted_array_start()`, `MPI_Irecv` and `MPI_Isend` calls are posted to the background hardware queue, returning execution control immediately to the CPU. In a fully optimized solver, the CPU could compute stencil updates for **inner domain nodes** (which don't depend on ghost cells) while ghost communication transfers across the network. Calling `export_to_ghosted_array_finish()` (with `MPI_Waitall`) is only required before updating **boundary nodes** that depend on those ghost values.