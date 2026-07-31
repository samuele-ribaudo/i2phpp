# Oral Exam Prep — Q&A Across All Three Worksheets

---

# WORKSHEET 1: SIMD

**Q: What is SIMD and what problem does it solve?**
A: Single Instruction, Multiple Data — one instruction operates on several data elements simultaneously, using wide hardware registers (128/256/512-bit). It exploits data-level parallelism on a single core, without threads or multiple processes.

**Q: What's the difference between `fixed_size_simd<T,N>` and `native_simd<T>`?**
A: `native_simd<T>` uses whatever width the compiler considers best for `T` on the current hardware (determined by `-march` flags). `fixed_size_simd<T,N>` forces exactly `N` lanes regardless of hardware — portable, but the compiler may need to internally split it into multiple native-width chunks (or use a scalar fallback) if `N` doesn't match a native register size.

**Q: Why is `N` a template parameter on `LaVector<number,N>` rather than a constructor argument?**
A: SIMD width must be known at compile time — `fixed_size_simd<T,N>` requires `N` as a compile-time constant. Making it a template parameter lets the compiler generate fully specialized code per width, and lets you instantiate/benchmark multiple widths (`LaVector<double,2>`, `<double,4>`, etc.) as genuinely different types in the same program.

**Q: What's the difference between `element_aligned` and `vector_aligned`?**
A: `element_aligned` tells the load/store to assume no more than normal per-element alignment (safe on any pointer). `vector_aligned` is a *promise* that the pointer is aligned to the full SIMD register width — lets the compiler use faster aligned instructions, but is undefined behavior if the promise is false. `LaVector`/`LaMatrix` use `std::vector`-backed heap storage, which is never guaranteed to be vector-aligned, so `element_aligned` is the only correct choice throughout.

**Q: Why does `LaVector` compute `ub`/`remainder` before the main loop?**
A: `ub = size - (size % N)` is the largest index reachable with full-width SIMD chunks; `remainder = size % N` is how many leftover elements don't fill a full chunk. This splits the loop into a full-speed SIMD portion and a small leftover portion needing separate handling.

**Q: Explain the masked-remainder pattern.**
A: Build a `simd_mask` with the first `remainder` lanes `true`, rest `false`. Zero-initialize the SIMD register (`x_simd(0)`) so untouched lanes are a safe, neutral value. Use `stdx::where(mask, x_simd).copy_from(...)` — this performs a genuine masked memory read, only touching real data for `true` lanes, never reading past the end of the array. Arithmetic then runs unmasked (safe because "fake" lanes hold 0, the identity for `+`), and the final store is masked again so nothing gets written out of bounds.

**Q: Why does `operator+=` need a masked *store* but `dot_product`'s remainder only needs a masked *load*?**
A: `operator+=` writes results back into memory (`data`), so writing unmasked would corrupt memory past the array's end. `dot_product` only accumulates into a register and never writes to memory — so only the *read* side needs protection against going out of bounds.

**Q: In your original submission, which required method was NOT vectorized?**
A: `dot_product` — it used a plain scalar loop. It should have mirrored `l2_norm_squared`'s pattern (SIMD accumulator, `stdx::reduce` once at the end), since a dot product is mathematically the same operation, just between two different vectors instead of one vector against itself.

**Q: Why keep the accumulator in a SIMD register and reduce only once at the end, instead of reducing every chunk?**
A: Horizontal reduction (`stdx::reduce`) costs several internal shuffle/add steps. Reducing every chunk pays that cost `size/N` times; reducing once at the end pays it exactly once, regardless of vector size.

**Q: For matrix-matrix multiplication, why is the loop order `row → k → col` instead of the naive `row → col → k`?**
A: `LaMatrix` stores data row-major (`std::vector<std::vector<number>>`), so a single row is contiguous but a column is not (each row is a separate heap allocation). The naive order needs `B[k][col]` for varying `k` — a non-contiguous column walk, unvectorizable. The `row→k→col` order instead loads a whole *row* of B at fixed `k`, which is contiguous, and accumulates that scaled row into the output row — this only works because computing `C = A×B` is equivalent to "row i of C = sum over k of A[i][k] × (row k of B)".

**Q: Why does matrix-matrix multiplication need to re-load `result` from memory on every `idx` iteration, instead of keeping an accumulator in a register the whole time?**
A: Because `idx` is the *middle* loop (between `row` and `col`), not the innermost one wrapping a single accumulator — the running total genuinely lives in `result`'s memory between iterations. This could be improved by swapping `col_idx` and `idx` so a register-based accumulator sums all `idx` contributions before writing to memory once per chunk — a known inefficiency worth flagging if asked to improve the code.

**Q: What happens if you use `fixed_size_simd<double,16>` on hardware whose native width is only 4?**
A: The compiler splits it into multiple native-width chunks internally — e.g. 4 separate 256-bit `vaddpd` instructions for N=16 on an AVX2 machine — completely transparently, at compile time, with zero code needed from you.

**Q: What happens for a remainder that doesn't divide evenly, e.g. `fixed_size_simd<double,17>` on native width 4?**
A: Compiler-dependent. On GCC/AVX2 (verified empirically): 4 full native chunks handle the first 16 elements via real `vaddpd`, and the 1 leftover element falls back to a genuine scalar `vaddsd`/`vmovsd`. For a remainder of 3 (not 1), the same compiler instead built a *padded* 4-wide register via a stack scratch buffer (gathering 3 real values + 1 zero), did ONE vectorized add, then stored back only the 3 real lanes — a different strategy chosen because it was cheaper for 3 elements than for 1.

**Q: Does using a wider-than-native SIMD width make code proportionally much slower?**
A: No — empirically, running identical logic at N=4 vs N=2 (native) on the same array size showed roughly 1.5–2× cost, not orders of magnitude. Large discrepancies in real benchmarks usually point to something else (build optimization level, debug assertions, benchmark measurement noise) rather than the chunk-splitting mechanism itself.

**Q: Why does the mask-building loop (`for i<remainder: mask[i]=true`) not cost anything noticeable at runtime?**
A: Since `N` is a compile-time constant, `remainder` can only take values `0..N-1` — a small, fully enumerable range. The compiler unrolls/eliminates the loop entirely, replacing it with a handful of compare/branch instructions or direct constant returns — verified by inspecting generated assembly (no loop instructions present).

**Q: What is the "favorable" relationship between cache line size and SIMD vector width?**
A: When the SIMD register width in bytes equals the cache line size (both commonly 64 bytes with AVX-512), every cache-line fetch is immediately and fully consumed by exactly one SIMD load — no wasted fetches, no instruction needing to straddle two separate cache-line fetches.

**Q: What is `trace()` and why is it NOT vectorized?**
A: Sum of diagonal elements (`A[i][i]` for all `i`) — only defined for square matrices. Not vectorizable because diagonal elements are never contiguous in memory (each is in a different row, a different heap allocation) — there's no way to SIMD-load several diagonal elements at once given row-major storage. This is a genuine structural limitation, not an oversight (unlike `dot_product`).

**Q: What are L1, L2, and L∞ norms, and how do their SIMD reductions differ?**
A: L1 = sum of absolute values (`reduction +`); L2 = sqrt of sum of squares (`l2_norm_squared` uses `reduction +`, then `norm_l2` takes the sqrt); L∞ = the single largest absolute value (`reduction max`/`hmax`, not a sum). L∞ needs a fundamentally different combining operation (max vs. sum), which is why its SIMD code structurally differs from the other two.

---

# WORKSHEET 2: THREADS & OPENMP

## Threads Fundamentals

**Q: What's the difference between `.join()` and `.detach()`?**
A: `.join()` blocks the calling thread until the target thread finishes — safe, deterministic. `.detach()` lets the thread run fully independently with no further tracking — if the program exits before a detached thread finishes, it's simply killed mid-execution, with no way to know or prevent this.

**Q: What does `std::jthread` add over `std::thread`?**
A: Automatic joining in its destructor (RAII) — you can't forget to join, and the program won't crash from an un-joined thread going out of scope.

**Q: What is a race condition, precisely?**
A: `++counter` is not atomic — it's read, add 1, write back. If two threads interleave those three steps (e.g., both read the same old value before either writes back), one increment is silently lost. The final result becomes dependent on unpredictable thread scheduling.

**Q: If a race condition consistently produces the "correct" result in your testing, does that mean the code is safe?**
A: No. This is undefined behavior in the C++ standard — the compiler/hardware may legally do anything, including consistently producing the mathematically correct answer (e.g., an optimizer may collapse `for(...)  ++counter;` into one `counter += N`, shrinking the interleaving window so much the race rarely manifests). It's still not guaranteed to stay correct across different compilers, optimization levels, or hardware.

**Q: What's the difference between `std::mutex::lock()/unlock()` and `std::lock_guard`?**
A: Functionally identical protection, but `lock_guard` locks on construction and unlocks automatically on destruction (RAII) — safe even with early returns or exceptions, which could cause a manual `unlock()` to be skipped, permanently locking the mutex.

**Q: How does a deadlock happen with two mutexes?**
A: If Thread A locks `mtx1` then tries `mtx2`, while Thread B locks `mtx2` then tries `mtx1`, and both grab their first lock before either reaches for the second, each ends up waiting forever for a lock the other is holding. Fix: always acquire multiple locks in the same, consistent order everywhere (or use `std::scoped_lock` for multiple mutexes at once).

**Q: At what exact point does a thread block on a mutex?**
A: Only at the specific `.lock()` call for that specific mutex, and only if that mutex happens to already be held by another thread at that exact moment. A thread never "looks ahead" to locks it'll need later — it only checks whichever mutex it's currently trying to acquire, right now.

**Q: What is a condition variable for, and why not just poll a flag in a loop?**
A: Lets a thread sleep efficiently until explicitly signalled, instead of wasting CPU repeatedly checking (busy-waiting). `cv.wait(lock, predicate)` atomically releases the lock and sleeps; on `notify_one()/notify_all()`, it re-acquires the lock and re-checks the predicate (guarding against spurious wakeups) before actually proceeding.

**Q: Why does `cv.wait` need `std::unique_lock` instead of `std::lock_guard`?**
A: `cv.wait` must unlock the mutex while sleeping (so other threads can proceed) and re-lock it before returning. `lock_guard` can only unlock once, on destruction — it has no "unlock now, relock later" capability. `unique_lock` supports exactly that.

## OpenMP

**Q: What does `#pragma omp parallel` do, exactly?**
A: Spawns N threads, and EVERY thread runs the ENTIRE following block — it does not automatically split work. Splitting requires `#pragma omp for` inside it.

**Q: What does `#pragma omp critical` do, and how does it compare to a mutex?**
A: Ensures only one thread executes the following statement/block at a time — functionally identical to wrapping it in `mtx.lock()/unlock()`, except OpenMP manages the hidden lock for you. By default all unnamed `critical` sections share one global lock (same over-restriction issue as one mutex protecting unrelated variables) — named critical sections (`critical(name)`) create independent locks.

**Q: What's the difference between `#pragma omp single` and `#pragma omp critical`?**
A: `critical` lets every thread execute the block, one at a time (runs N times total). `single` lets exactly ONE thread execute the block at all — the others skip it entirely and wait at an implicit barrier.

**Q: Explain `schedule(static)` vs `schedule(dynamic)` vs `schedule(guided)`.**
A: `static`: chunks assigned once, upfront, round-robin — zero runtime overhead, but no load balancing if some iterations cost more than others. `dynamic`: chunks handed out on-demand as threads finish — best load balancing, but real per-chunk coordination overhead. `guided`: starts with large chunks, shrinks over time — a middle ground. Use `static` when work per iteration is uniform (no rebalancing needed); `dynamic`/`guided` only pay off when iteration cost is genuinely uneven.

**Q: What does `reduction(+ : var)` actually do under the hood?**
A: Gives each thread its own private copy of `var`, initialized to the operator's identity (0 for `+`), lets each thread accumulate independently with zero synchronization, then combines all private copies into the original variable once, after the loop — avoiding the race condition that would occur if every thread wrote directly to one shared `var`.

**Q: What does `collapse(2)` require, and why?**
A: The two loops must be "perfectly nested" (nothing between the outer loop's header and the inner loop's start) and have rectangular bounds (inner bound not dependent on the outer variable). This lets OpenMP flatten them into one combined iteration count and distribute that flattened space across threads — useful when the outer loop alone doesn't have enough iterations to use all requested threads.

**Q: Why is matrix-matrix multiplication NOT collapsed across all three loops, only the outer one?**
A: Collapsing all three would let multiple threads land on the same output cell (different `idx`/`k` values, same `row`/`col`) and race on the `+=` accumulation. Splitting only the outermost `row_idx` loop gives each thread an exclusive set of output rows — no cell is ever touched by two threads.

**Q: What does `#pragma omp ordered` guarantee, and how does it differ from `critical`?**
A: Forces a block to execute in the loop's original iteration order (i=0 before i=1 before i=2...), regardless of which thread runs which iteration or how fast each thread finishes other work — `critical` only guarantees one-at-a-time, in whatever order threads happen to arrive.

**Q: Explain `shared`, `private`, `firstprivate`, `lastprivate`.**
A: `shared`: one object, every thread sees the same memory, needs protection against concurrent writes. `private`: each thread gets its own copy, starting UNINITIALIZED; changes never escape the region. `firstprivate`: like private, but each copy starts PRE-FILLED with the outside value; changes still don't escape. `lastprivate`: like private, but the value from the LOGICALLY LAST loop iteration (not whichever thread finishes last in real time) is copied back to the outer variable after the loop.

**Q: Why use `default(none)`?**
A: Forces every variable referenced inside the parallel region to be explicitly classified. Without it, OpenMP's actual default is mostly `shared` — exactly how races sneak in unnoticed. `default(none)` is a safety net that makes data-sharing intent explicit and compiler-checked.

## Thread Pool

**Q: Why build a thread pool instead of spawning a new `std::thread` per task?**
A: Thread creation/destruction has real OS overhead. A pool creates a fixed set of worker threads once, and they stay alive, repeatedly pulling new jobs from a queue — avoiding that overhead on every single task submission (critical for something like a heat solver's millions of time-step-level tasks).

**Q: Why must the lock be released BEFORE calling `job()` inside the worker loop?**
A: If the lock were held while a job runs, only one worker could ever execute a job at a time — completely defeating the purpose of having multiple workers. The lock is only needed briefly, to safely pop a job off the shared queue.

**Q: Why does `enqueue` use `notify_one()` but the destructor uses `notify_all()`?**
A: `enqueue` added exactly one new job — waking every sleeping worker would cause all-but-one to immediately find the queue empty and go back to sleep (wasted wakeups). Shutdown needs `notify_all()` because EVERY worker must wake up and check the stop flag to actually exit.

**Q: Why track `active_jobs` separately from queue emptiness?**
A: "Popped off the queue" and "finished executing" are different moments. `wait_for_all()` needs both `jobs.empty()` AND `active_jobs==0` to know all work has truly completed — checking queue emptiness alone would return early while a worker is still mid-job.

**Q: Why two separate condition variables (`queue_condition` and `work_done_condition`)?**
A: They serve different audiences: `queue_condition` wakes idle WORKERS when new work arrives (or shutdown is signaled). `work_done_condition` wakes whoever called `wait_for_all()` once everything is truly finished. Different signals, different listeners — combining them into one would be confusing and error-prone.

## Linear Algebra + Heat Solver with OpenMP

**Q: Why is `dot_product` correctly parallelized with OpenMP but was left unvectorized in the SIMD (Worksheet 1) version?**
A: In Worksheet 2, `reduction(+ : result)` makes thread-level reduction essentially free to add correctly. In Worksheet 1, the vectorized version simply wasn't implemented for that method — a gap, not a technical obstacle (the fix mirrors `l2_norm_squared`'s existing SIMD pattern).

**Q: Why does `current_solution.swap(old_solution)` matter for correctness under OpenMP parallelization, not just performance?**
A: The stencil computation reads NEIGHBORING values. If threads updated `current_solution` in place while other threads read neighboring values from the same array, a thread could read an already-updated neighbor value instead of the value from the start of this time step — silently corrupting the numerical scheme (turning parallel Jacobi/explicit-Euler into an inconsistent race). Double buffering (write only to a separate `current_solution`/`x_new`, read only from `old_solution`/`x_old`) guarantees every thread sees a consistent, unmodified snapshot of neighboring data.

**Q: Why is `schedule(static)` the right choice for system matrix assembly?**
A: Every interior row does the same fixed amount of work (same-length inner loop over dimensions, no data-dependent branching cost). Uniform per-iteration cost means nothing to load-balance — `dynamic`/`guided` would only add overhead with no benefit.

**Q: What's the difference between CPU time and wall-clock time in a parallel benchmark?**
A: Wall-clock = real elapsed time from start to end. CPU time = sum of active execution time across all cores used. An ideal N-thread region running 1 second of wall-clock time would show N seconds of CPU time.

---

# WORKSHEET 3: MPI

## Fundamentals

**Q: Contrast threads/processes: shared vs. distributed memory.**
A: Threads share one heap (same process), with separate stacks — communication is direct memory access, protected by mutexes. MPI processes have fully isolated address spaces — one process cannot see another's memory at all; the ONLY way data moves is explicit message passing over the network (even on the same machine).

**Q: What are rank, size, and communicator?**
A: `size` = total number of processes in a communicator (fixed, same answer on every process). `rank` = this specific process's unique ID (0 to size-1, different per process). `communicator` (e.g. `MPI_COMM_WORLD`) = a defined group of processes that can communicate with each other. All processes run the identical compiled program (SPMD); they behave differently only because `rank` differs per process.

**Q: Blocking vs non-blocking point-to-point communication?**
A: `MPI_Send`/`MPI_Recv` block until the operation completes. `MPI_Isend`/`MPI_Irecv` return immediately, letting the process continue other work while communication happens in the background; completion is confirmed later via `MPI_Wait`/`MPI_Waitall`/`MPI_Test`. Non-blocking allows computation to overlap with (hide) communication latency.

**Q: Why must a buffer passed to `MPI_Isend`/`MPI_Irecv` never be modified before the request completes?**
A: The operation may still be reading from (send) or writing into (receive) that memory in the background — touching it early causes a race between your own code and the in-flight MPI transfer, corrupting the data.

**Q: What is Eager vs Rendezvous protocol?**
A: For small messages, MPI eagerly sends data immediately, buffering it on the receiver side even before a matching `Recv` is posted (Eager protocol) — low latency but uses extra memory. For large messages, MPI first does a handshake to confirm the receiver is ready before sending the actual payload (Rendezvous) — avoids buffering overhead at the cost of extra round-trip latency.

## Ghost Cells / Halo Exchange

**Q: What is a ghost element?**
A: A read-only local copy of a value actually OWNED by a neighboring process — needed because a stencil computation near a partition boundary requires neighboring values that live in a different process's private memory.

**Q: Why does `dot_product` sum only `locally_owned_size()` elements, excluding ghosts?**
A: Ghost elements are copies of another process's owned data. If every process summed its ghosts too, boundary-adjacent values would be double-counted (once by the true owner, once by the neighbor holding a ghost copy) — giving a wrong global total.

**Q: Why do elementwise operators (`+=`, `-=`, `*=`) operate over ALL of `local_elements`, including ghosts, unlike `dot_product`?**
A: This is safe as long as `update_ghost_values()` is always called again before ghosts are next read — any arithmetic done on the ghost region becomes stale/meaningless but gets overwritten by the next refresh. It IS some wasted computation on values that don't semantically belong to this process, and is a slight inconsistency worth naming if pushed on it.

**Q: What is `std::span` and why use it in `update_ghost_values`?**
A: A lightweight, non-owning view (pointer + length) into an existing contiguous array — no allocation, no copying. `local_elements` is split into two spans (owned region, ghost region) purely by pointer arithmetic, letting the Partitioner operate on "just the owned part" and "just the ghost part" as if separate, without any real separate storage.

## The Partitioner (`determine_communication_pattern`)

**Q: Explain the three steps of `determine_communication_pattern`.**
A: (1) Ghost request phase — every process broadcasts its full list of needed ghost indices to everyone via `MPI_Allgather`(counts)+`MPI_Allgatherv`(actual indices), so every process ends up holding everyone's full wishlist. (2) Ownership check — purely local: each process scans that combined wishlist against its OWN `locally_owned_indices`, building `export_targets_and_indices` (who wants data I own). (3) Transpose — since knowing who I export to doesn't tell me who imports FROM me, sizes are exchanged via `MPI_Alltoall`, then the actual index lists are exchanged via non-blocking `Isend`/`Irecv`, producing `import_targets_and_indices`.

**Q: What's the difference between `MPI_Allgather` and `MPI_Alltoall`?**
A: `Allgather`: every process broadcasts the SAME single value to EVERYONE — all processes end up with an identical combined list. `Alltoall`: every process sends a DIFFERENT, personalized value to EACH other process individually — like transposing a matrix, where row i (what rank i sends to everyone) becomes column i (what everyone sent to rank i) on the receiving side.

**Q: In `MPI_Alltoall(send_sizes.data(), 1, MPI_INT, ...)`, why is the count argument 1 even though multiple indices might be exchanged?**
A: This particular call only exchanges COUNTS (one integer per destination rank), not the actual index data. The count argument describes how many elements go to EACH INDIVIDUAL rank per call, not the grand total — since exactly one integer is sent per rank, that's 1. The actual variable-length index lists are exchanged afterward, separately, via `Isend`/`Irecv`, using the sizes this call just determined.

**Q: Is there a loop involved in an `MPI_Alltoall` call, even though you write no loop in your code?**
A: Not in your own source — you call it once, and every rank in the communicator calls that same line simultaneously. Internally, the MPI library implementation performs the actual cross-process data shuffling; array POSITION conventionally maps to destination/source rank (index r = data for/from rank r), with no explicit indexing needed in your code.

**Q: When you `MPI_Waitall` on a batch of requests, does the process wait for OTHER ranks' requests too?**
A: No — each rank's `Waitall` only waits on the requests IT ITSELF created (its own posted sends/receives). Different ranks can unblock at entirely different real times, independently — there's no implicit synchronization forcing them to finish together (that would require an explicit `MPI_Barrier`).

**Q: Why use the global index as the message tag in ghost exchange?**
A: Non-blocking transfers can complete out of order. Using the unique global index as the tag guarantees an incoming message maps unambiguously to the correct destination memory location, regardless of the order messages actually arrive in.

## DoF Handler / Domain Decomposition

**Q: How is the domain split across ranks?**
A: Always along the LAST spatial dimension (`n_dofs[dim-1]`) — 1D: contiguous chunks; 2D: horizontal strips (full x-width); 3D: full xy-slabs. Chosen because the array is stored with the last dimension varying slowest and earlier dimensions fastest — splitting along the last dimension keeps each rank's owned data genuinely contiguous in the flat array.

**Q: How is the oversubscription problem (more ranks than slices) handled?**
A: `MPI_Comm_split` creates a smaller sub-communicator: ranks with `rank < max_slices` get `color=0` (active, grouped together); the rest get `color=MPI_UNDEFINED`, receive back `MPI_COMM_NULL`, initialize an empty vector, and return immediately, sitting idle for the rest of the simulation.

**Q: How is the "at most 1 element difference" load balancing achieved when the split doesn't divide evenly?**
A: `remainder = n_dofs[dim-1] % size` leftover slices are distributed one each to the first `remainder` ranks via `begin_offset`/`end_offset`, which shift each rank's starting/ending point in the index range by however many bonus slices have already been allocated to earlier ranks — so the actual row count owned is `local_length + (end_offset − begin_offset)`, giving early ranks one extra slice and later ranks the base amount only.

**Q: Why does each rank compute its own range independently, with no communication needed for Task 3's splitting (unlike Task 2's ghost pattern)?**
A: Every rank already knows `rank`, `size`, and the grid dimensions from the start — the arithmetic is a pure, deterministic function of those values, so no rank needs information only another rank has. Task 2's ghost determination genuinely needed communication because ownership of arbitrary requested indices isn't derivable from local knowledge alone.

## Heat Operator with MPI

**Q: Why must `update_ghost_values()` be called before every stencil evaluation in `advance_time_step`?**
A: The stencil needs neighboring values; some neighbors live in adjacent processes' memory as ghost copies. Without refreshing first, a rank would compute using STALE (previous time step's) ghost data — silently wrong results, not a crash.

**Q: How does the loop bound differ between the OpenMP (Worksheet 2) and MPI (Worksheet 3) versions of `advance_time_step`?**
A: OpenMP looped over the ENTIRE global domain (`i < n_global_dofs`) since every thread could see the whole shared array. MPI loops only over `locally_owned_size()` — each rank only ever touches its own private slice, since nothing else is reachable in its memory at all.

**Q: Why does the implicit/Jacobi residual computation need `MPI_Allreduce`, similar to `dot_product`?**
A: Convergence ("have we solved the system yet") is a property of the WHOLE domain, not any single rank's slice. Each rank computes its own local residual contribution; `Allreduce` combines them into one global residual norm, delivered identically to every rank, so every rank makes the same, consistent convergence decision.

**Q: What does `MPI_IN_PLACE` do?**
A: Tells `Allreduce` to read the input from and write the combined result back into the SAME variable, avoiding the need for a separate temporary send buffer when the original value isn't needed after the reduction.

**Q: Why must every process call a collective operation (like `Allreduce`/`Allgather`/`Alltoall`), even ones with no data to contribute?**
A: Collectives are synchronization points requiring participation from every member of the communicator — if even one process never calls it, every other process blocks forever waiting for a contribution that will never come. This is also why idle (oversubscribed) ranks are cleanly excluded via `MPI_Comm_split`/`MPI_COMM_NULL` rather than just skipping collective calls ad hoc.

---

# Cross-Worksheet Big-Picture Questions

**Q: How does OpenMP threading relate to SIMD — does parallelizing with threads also vectorize automatically?**
A: No — they're separate, orthogonal mechanisms. `#pragma omp parallel for` splits loop iterations across CPU CORES (threads) — MIMD, across-core parallelism. SIMD vectorizes WITHIN one thread's own instruction stream, using that core's wide registers — separate from and not implied by threading. OpenMP has its own explicit SIMD directive (`#pragma omp simd`) distinct from `#pragma omp for`. Peak throughput = (# cores used) × (SIMD width per core) × (clock speed) — both layers must be active together for maximum performance.

**Q: Trace the "parallelism ladder" across all three worksheets.**
A: Worksheet 1 (SIMD): single-core, data-level parallelism, one instruction/many array elements, shared memory trivially (it's all one thread). Worksheet 2 (OpenMP/threads): multi-core, shared memory — every thread can see the same variables, so races/mutexes/condition variables matter. Worksheet 3 (MPI): multi-node, distributed memory — no shared memory at all; every piece of cross-process data movement is an explicit message, and "who owns what" must be tracked and communicated deliberately (ghost cells, partitioners).

**Q: What's the common thread (no pun intended) in how "the remainder problem" is handled across all three worksheets?**
A: SIMD: masked/scalar handling of `size % N` leftover elements. OpenMP: threads' index ranges computed with `total/N_threads`, remainder distributed to first few threads. MPI: same remainder-distribution formula, applied to distributing grid slices across ranks (`begin_offset`/`end_offset` giving the first `remainder` ranks one bonus slice). Same fundamental "integer division has leftovers" problem, solved with structurally similar logic at every layer of parallelism.
