
#pragma once

#include <i2phpp/util/index_set.hpp>
#include <i2phpp/util/partitioner.hpp>
#include <mpi.h>

#include <vector>

namespace i2phpp
{
  class ParallelDistributedVector
  {
  public:
    /**
     * @brief Constructs a vector using only the MPI communicator. No memory is
     * allocated; intended for later initialization.
     *
     * @param communicator The MPI communicator to associate with the vector.
     */
    explicit ParallelDistributedVector(MPI_Comm communicator);

    /**
     * @brief Constructs and initializes a distributed vector using a partitioner.
     * Allocates memory for locally owned and ghost entries.
     *
     * @param partitioner The partitioning information for the vector layout.
     * @param initial_value Value used to initialize all entries.
     */
    explicit ParallelDistributedVector(const Partitioner &partitioner,
                                       double             initial_value = 0.);

    /**
     * @brief Constructs a vector with a specified range of locally owned indices.
     * Ghost entries are not included.
     *
     * @param local_range IndexSet representing locally owned indices.
     * @param communicator MPI communicator for parallel distribution.
     * @param initial_value Value used to initialize all entries.
     */
    ParallelDistributedVector(const IndexSet &local_range,
                              MPI_Comm        communicator,
                              double          initial_value = 0.);

    /**
     * @brief Constructs a vector with both locally owned and ghost indices.
     * Initializes all entries with the provided value.
     *
     * @param local_range IndexSet of locally owned indices.
     * @param ghost_indices IndexSet of ghost indices.
     * @param communicator MPI communicator for data distribution.
     * @param initial_value Value used to initialize all entries.
     */
    ParallelDistributedVector(const IndexSet &local_range,
                              const IndexSet &ghost_indices,
                              MPI_Comm        communicator,
                              double          initial_value = 0.);

    /**
     * Reinitializes the vector with a new index set and MPI communicator.
     *
     * @param locally_owned_indices List of locally owned global indices.
     * @param ghost_indices List of locally relevant ghost global indices.
     * @param initial_value The value with which both locally owned and ghost
     * elements shall be initialized.
     * @param communicator New MPI communicator.
     */
    void
    reinit(const IndexSet &locally_owned_indices,
           const IndexSet &ghost_indices,
           double          initial_value = 0.,
           MPI_Comm        communicator  = MPI_COMM_WORLD);

    /**
     * Return the locally available element with local index @p local_index. The
     * elements [0, locally_owned_size()) are locally owned elements and
     * elements with local indices [locally_owned_size(), locally_owned_size()+
     * partitioner.n_ghost_indices()) are ghost elements.
     *
     * @param local_index Local index of the element of interest.
     * @return Element at the given local index.
     *
     * @throw MPI_Abort If local index is out of range this means greater than
     * the size of locally available indices.
     */
    double &
    local_element(unsigned local_index);

    /**
     * Same as above but returns a constant reference.
     */
    [[nodiscard]] const double &
    local_element(unsigned local_index) const;

    /**
     * Return a reference to the element with the global index @p global_index.
     * This function assumes that the corresponding element is either within the
     * locally owned or ghost range of elements. If this is not the case it
     * throws. Due to this assumption no MPI communication is performed when
     * calling this function.
     */
    double &
    global_element(unsigned global_index);

    /**
     * Same as above but returns a constant reference.
     */
    [[nodiscard]] const double &
    global_element(unsigned global_index) const;

    /**
     * Return the value at the index @p global_index, independent of weather the
     * element is stored locally (either local owned or as ghost value) or on
     * another MPI process. The function takes care of identifying the owning
     * process and establishing all the communication required to broadcast the
     * value to the other processes in the communicator.
     *
     * If the index of the desired value is owned by the current MPI process it
     * broadcasts it to all other processes in the respective communicator. If
     * it is not owned it will receive the value by the broadcasting process.
     *
     * @param global_index The global index of the element of interest.
     * @return Vector value at the given global index.
     *
     * @note This is a collective operation, i.e., it must be called by all
     * processes in the corresponding MPI communicator as otherwise the function
     * call results in a deadlock.
     *
     * @note A call to this function is computationally expensive as it requires
     * significant communication and synchronization issues. Only use it rarely!
     */
    [[nodiscard]] double
    gather_global_element(unsigned global_index) const;

    /**
     * Return the size, i.e., the number, of locally owned elements.
     */
    [[nodiscard]] unsigned
    locally_owned_size() const;

    /**
     * Return the number of ghost elements.
     */
    [[nodiscard]] unsigned
    ghost_size() const;

    /**
     * Return the global size of the distributed vector, i.e., the accumulated
     * number of locally owned elements on all processes.
     */
    [[nodiscard]] unsigned
    global_size() const;

    /**
     * Updates the ghost values. This is a collective operation and needs to be
     * called by all MPI processes.
     */
    void
    update_ghost_values();

    /**
     * Swaps the data of this vector with another vector using the swap
     * function of the underlying data structure. This means it swaps all
     * locally relevant elements stored in the vector.
     *
     * @param other The object whose data is to be swapped with this object.
     *
     * @throws MPI_Abort if the two objects are not compatible.
     */
    void
    swap(ParallelDistributedVector &other);

    /**
     * Adds the given vector @p vec to the current vector and returns the
     * resulting vector.
     *
     * @param vec The vector to add.
     * @return The sum of the current vector and @p vec.
     *
     * @throw MPI_Abort If the two vectors are not compatible.
     */
    ParallelDistributedVector
    operator+(const ParallelDistributedVector &vec) const;

    /**
     * Adds the given vector @p vec to the current vector, modifying the
     * current vector.
     *
     * @param vec The vector to add.
     *
     * @throw MPI_Abort If the two vectors are not compatible.
     */
    void
    operator+=(const ParallelDistributedVector &vec);

    /**
     * Subtracts the given vector @p vec from the current vector and returns the
     * resulting vector.
     *
     * @param vec The vector to subtract.
     * @return The difference between the current vector and @p vec.
     *
     * @throw MPI_Abort If the two vectors are not compatible.
     */
    ParallelDistributedVector
    operator-(const ParallelDistributedVector &vec) const;

    /**
     * Subtracts the given vector @p vec from the current vector, modifying the
     * current vector.
     *
     * @param vec The vector to subtract.
     *
     * @throw MPI_Abort If the two vectors are not compatible.
     */
    void
    operator-=(const ParallelDistributedVector &vec);

    /**
     * Multiplies each element of the current vector by the scalar @p num and
     * returns the resulting vector.
     *
     * @param num The scalar to multiply with.
     * @return The scaled vector.
     */
    ParallelDistributedVector
    operator*(double num) const;

    /**
     * Multiplies each element of the current vector by the scalar @p num
     * modifying the current vector.
     *
     * @param num The scalar to multiply with.
     */
    void
    operator*=(double num);

    /**
     * Compute the euclidean norm of the vector.
     *
     * @return Euclidean norm of the vector.
     */
    [[nodiscard]] double
    l2_norm() const;

    /**
     * Returns the dot product of the current vector and the provided @p other
     * vector.
     *
     * @param other The vector to compute the dot product with.
     * @return The dot product of the two vectors.
     *
     * @throw MPI_Abort If the two vectors are not compatible.
     */
    [[nodiscard]] double
    dot_product(const ParallelDistributedVector &other) const;

    /**
     * Returns whether this vector is compatible with @p other. For two vectors
     * being compatible the partitioners of both vectors must be compatible.
     *
     * @param other Object to compare compatibility with.
     */
    [[nodiscard]] bool
    is_compatible(const ParallelDistributedVector &other) const;

    /**
     * Return the partitioner used by the parallel distributed vector.
     */
    [[nodiscard]] const Partitioner &
    get_partitioner() const;

    /**
     * Return the communicator used by the parallel distributed vector.
     */
    [[nodiscard]] MPI_Comm
    get_communicator() const;

  private:
    /**
     * Check if this vector is compatible with @p other. For two vectors being
     * compatible the partitioners of both vectors must be compatible. If the
     * vectors are not compatible this function throws.
     *
     * @param other Object to compare compatibility with.
     *
     * @throw MPI_Abort If the vectors are not compatible.
     */
    void
    assert_compatibility(const ParallelDistributedVector &other) const;

    /// Storage for the locally relevant elements. The elements [0,
    /// locally_owned_size()) are locally owned elements and elements with local
    /// indices [locally_owned_size(), locally_owned_size()+
    /// partitioner.n_ghost_indices()) are ghost elements.
    std::vector<double> local_elements = {};

    /// Partitioner taking care of MPI communication and handling indices of
    /// locally owned and ghost elements.
    Partitioner partitioner;

    /// MPI communicator used by the distributed vector and its partitioner.
    MPI_Comm communicator;
  };
} // namespace i2phpp