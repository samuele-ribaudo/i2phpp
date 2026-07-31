# OpenMP & MPI Command Reference

---

# OpenMP Directives & Clauses

## `#pragma omp parallel`
Spawns a team of threads; **every** thread executes the entire following block (no automatic work-splitting).

```cpp
#pragma omp parallel num_threads(4) default(none) shared(a, b) private(c) firstprivate(d)
```
| Clause | Meaning |
|---|---|
| `num_threads(n)` | How many threads to spawn for this region. Omit to use the OpenMP runtime default (usually all available cores). |
| `default(none)` | Forces every variable used inside to be explicitly classified below — compiler error otherwise. Prevents accidental implicit-shared races. |
| `shared(a, b)` | These variables are ONE object, same memory, seen/modified by all threads — needs protection (`critical`/`reduction`/`atomic`) against concurrent writes. |
| `private(c)` | Each thread gets its own independent copy, starting **uninitialized**. Changes never propagate back outside the region. |
| `firstprivate(d)` | Each thread gets its own copy, but **pre-initialized** with the value from outside the region. Changes still don't escape. |

## `#pragma omp for`
Placed above a plain `for` loop **inside** a `parallel` region — splits that loop's iterations across the already-existing threads.

```cpp
#pragma omp for schedule(dynamic, 10) reduction(+ : sum) collapse(2) nowait lastprivate(x) ordered
```
| Clause | Meaning |
|---|---|
| `schedule(static, n)` | Fixed-size chunks of `n` iterations, assigned once upfront, round-robin. Zero runtime overhead; no load balancing. Best when iteration cost is uniform. |
| `schedule(dynamic, n)` | Chunks of `n`, handed out on-demand as threads finish their current chunk. Best load balancing; real per-chunk coordination overhead. |
| `schedule(guided, n)` | Starts with large chunks, shrinks toward `n` as the loop progresses. Middle ground between `static` and `dynamic`. |
| `reduction(op : var)` | `op` = `+`, `-`, `*`, `max`, `min`, etc. Each thread gets a private copy of `var`, initialized to `op`'s identity value, accumulates independently, then all copies are combined into `var` using `op` once the loop finishes. |
| `collapse(n)` | Flattens `n` perfectly-nested loops (nothing in between them, rectangular bounds) into one combined iteration space before splitting across threads. Needed when the outer loop alone has too few iterations to use all threads. |
| `nowait` | Removes the implicit barrier normally present at the end of the `for` loop — lets faster threads move on to subsequent code immediately instead of waiting for slower threads to finish this loop. Only safe if the next code doesn't depend on this loop's full completion. |
| `lastprivate(var)` | Like `private`, but the value written by whichever thread executed the loop's **logically last iteration** (not whichever finishes last in real time) is copied back to the outer `var` after the loop. |
| `ordered` | Declares that this loop will use `#pragma omp ordered` internally — required precondition for using that directive inside the loop body. |

## `#pragma omp parallel for`
Shorthand combining `#pragma omp parallel` + `#pragma omp for` into one line — used when the parallel region contains nothing but the one loop. Accepts clauses from both (`num_threads`, `schedule`, `collapse`, etc. together).

## `#pragma omp critical`
```cpp
#pragma omp critical
  ++counter;
// or, for multiple statements:
#pragma omp critical
{
  ++counter;
  ++other_var;
}
```
Ensures only **one thread at a time** executes the following statement (or `{ }` block) — like an automatic mutex `lock()`/`unlock()` around it. All unnamed `critical` sections in a program share **one global lock** by default (independent sections need `#pragma omp critical(name)` with distinct names, so unrelated critical sections don't block each other unnecessarily).

## `#pragma omp single`
```cpp
#pragma omp single
{ ... }
```
Exactly **one** thread (whichever arrives first) executes the block; the other threads **skip it entirely** and wait at an implicit barrier at its end. Different from `critical`, where every thread eventually runs the block, just one at a time.

## `#pragma omp barrier`
Forces every thread in the current parallel region to wait at this exact line until **all** threads have reached it, before any are allowed to proceed further. Note: the closing `}` of a `parallel` region has an implicit barrier automatically, without needing this directive.

## `#pragma omp ordered`
```cpp
#pragma omp ordered
{ ... }
```
(Requires the enclosing loop to be declared with the `ordered` clause.) Forces this block to execute strictly in the loop's original iteration order (`i=0` before `i=1` before `i=2`...) — regardless of which thread runs which `i`, or how fast each thread finishes its non-ordered work. Stronger guarantee than `critical` (which only guarantees one-at-a-time, in arrival order).

## Runtime API functions (not pragmas — real function calls)
| Function | Meaning |
|---|---|
| `omp_get_thread_num()` | Returns the calling thread's ID within its current team (0 to N-1). |
| `omp_get_num_threads()` | Returns how many threads exist in the current parallel region. |
| `omp_set_max_active_levels(n)` | Allows nested `parallel` regions up to `n` levels deep (nesting is off by default). |

---

# MPI Functions

## Setup / Teardown
```cpp
MPI_Init(&argc, &argv);
MPI_Finalize();
```
| Call | Meaning |
|---|---|
| `MPI_Init(&argc, &argv)` | Initializes the MPI environment. Must be called before any other MPI function. |
| `MPI_Finalize()` | Shuts down the MPI environment. No MPI calls are allowed after this. |
| `MPI_Abort(comm, errorcode)` | Immediately terminates all processes in `comm` with the given error code — used for unrecoverable setup errors (e.g., "need at least 2 processes"). |

## Identity
```cpp
MPI_Comm_rank(communicator, &rank);
MPI_Comm_size(communicator, &size);
```
| Argument | Meaning |
|---|---|
| `communicator` | Which group of processes you're asking about (e.g. `MPI_COMM_WORLD` = everyone). |
| `&rank` / `&size` | Output pointer — filled with this process's ID (`rank`, different per process) or the total process count (`size`, same on every process). |

## Point-to-Point: Blocking
```cpp
MPI_Send(&data, count, datatype, dest, tag, communicator);
MPI_Recv(&data, count, datatype, source, tag, communicator, &status);
```
| Argument | Meaning |
|---|---|
| `&data` | Pointer to the buffer being sent from / received into. |
| `count` | Number of elements (of `datatype`) in the buffer — NOT bytes. |
| `datatype` | e.g. `MPI_INT`, `MPI_DOUBLE`, `MPI_UNSIGNED` — the type of each element. |
| `dest` / `source` | Rank of the receiving/sending process. `MPI_ANY_SOURCE` on the receive side means "accept from whichever rank sends first." |
| `tag` | An integer label to disambiguate messages — a `Recv` only matches a `Send` with the same tag (and source, unless `MPI_ANY_SOURCE`). Worksheet 3 uses the global array index as the tag, so out-of-order arrivals still map to the correct destination. |
| `communicator` | The process group this send/receive happens within. |
| `&status` / `MPI_STATUS_IGNORE` | Output info about the received message (actual sender, tag, etc.) — `MPI_STATUS_IGNORE` if you don't need it. |

Both calls **block** — `Send` doesn't return until it's safe to reuse the buffer (may or may not mean the receiver has read it, depending on message size/protocol); `Recv` doesn't return until the expected data has actually arrived.

## Point-to-Point: Non-blocking
```cpp
MPI_Isend(&data, count, datatype, dest, tag, communicator, &request);
MPI_Irecv(&data, count, datatype, source, tag, communicator, &request);
MPI_Wait(&request, &status);
MPI_Waitall(count, requests_array, statuses_array_or_MPI_STATUSES_IGNORE);
MPI_Test(&request, &flag, &status);
```
| Argument | Meaning |
|---|---|
| `&request` | Output — a "ticket" representing this in-flight operation. The call returns immediately; the actual send/receive completes later, in the background. |
| `MPI_Wait(&request, ...)` | Blocks until this ONE specific request completes. |
| `MPI_Waitall(count, requests, ...)` | Blocks until ALL requests in the given array complete — doesn't matter which finishes first; returns once every one is done. |
| `MPI_Test(&request, &flag, ...)` | Non-blocking check — sets `flag` to true/false depending on whether the request has completed yet, without waiting. Lets you poll instead of blocking. |

**Rule:** never read from a send buffer or write to a receive buffer until its request has been confirmed complete via `Wait`/`Waitall`/`Test` — doing so races against the in-flight background transfer.

## Collective Operations
All processes in the communicator **must** call the same collective operation — if even one skips it, every other process blocks forever.

```cpp
MPI_Allreduce(&local_val, &global_val, count, datatype, op, communicator);
```
| Argument | Meaning |
|---|---|
| `&local_val` | This process's own input value(s) — the send buffer. |
| `&global_val` | Output — every process receives the SAME combined result here (unlike `MPI_Reduce`, which only delivers to one designated rank). |
| `count` | Number of elements being reduced. |
| `datatype` | e.g. `MPI_DOUBLE`, `MPI_INT`. |
| `op` | Combining operation: `MPI_SUM`, `MPI_MAX`, `MPI_MIN`, etc. |
| `MPI_IN_PLACE` | Can be passed as `&local_val` to mean "read from and overwrite `&global_val` directly" — avoids needing a separate temporary input buffer. |

```cpp
MPI_Allgather(&my_value, sendcount, sendtype, recv_array.data(), recvcount, recvtype, communicator);
```
Every process contributes ONE value; every process ends up with the FULL list of everyone's values (no combining — just collecting). `sendcount`/`recvcount` = how many elements each individual process contributes/receives (usually equal).

```cpp
MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts_array, displs_array, recvtype, communicator);
```
Variable-length version of `Allgather`, needed when different processes contribute different-sized lists.
| Argument | Meaning |
|---|---|
| `recvcounts_array` | Array — how many elements EACH rank is contributing (can differ per rank). |
| `displs_array` | Array — the starting offset in the combined output array where each rank's data should be placed (computed as a running sum of previous ranks' counts). |

```cpp
MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, communicator);
```
Every process sends a DIFFERENT, personalized chunk to EACH other process (a full "everyone talks to everyone" transpose). `sendcount`/`recvcount` = how many elements go to/come from **each individual** rank (not the total across all ranks) — since `sendbuf` is implicitly divided into `comm_size` chunks of this size, one per destination, by array position convention.

```cpp
MPI_Comm_split(old_comm, color, key, &new_comm);
```
| Argument | Meaning |
|---|---|
| `old_comm` | The communicator being split. |
| `color` | Processes with the SAME color end up grouped into the same new sub-communicator. `MPI_UNDEFINED` excludes a process entirely — it receives back `MPI_COMM_NULL`. |
| `key` | Controls the new rank ordering within each sub-group (usually pass the old `rank` to preserve relative order). |
| `&new_comm` | Output — the newly created (smaller) communicator for this process's group. |

```cpp
MPI_Barrier(communicator);
```
Blocks every process in `communicator` until **all** of them have reached this call — pure synchronization, no data exchanged.

## Special Constants
| Constant | Meaning |
|---|---|
| `MPI_COMM_WORLD` | The communicator containing every process launched. |
| `MPI_COMM_NULL` | Represents "not part of any communicator" — assigned to processes excluded via `MPI_Comm_split`. |
| `MPI_UNDEFINED` | Passed as a `color` to `Comm_split` to mean "exclude me from any group." |
| `MPI_ANY_SOURCE` | Used in `Recv` to accept a message from any sending rank. |
| `MPI_STATUS_IGNORE` / `MPI_STATUSES_IGNORE` | Tells MPI you don't need the status/statuses output — skips filling it in. |
| `MPI_IN_PLACE` | Used in collectives (e.g. `Allreduce`) to reuse the output buffer as the input, avoiding a separate temporary. |
