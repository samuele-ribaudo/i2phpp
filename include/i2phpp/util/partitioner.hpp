#pragma once

#include <i2phpp/util/index_set.hpp>
#include <mpi.h>

#include <ranges>
#include <utility>
#include <vector>

namespace i2phpp
{
  /**
   * @class Partitioner
   * @brief Manages partitioning and communication of distributed data across
   * MPI processes.
   *
   * This class provides functionality for managing locally owned and ghost
   * indices, mapping between global and local indices, and exporting data from
   * owned to ghosted regions. It also sets up the communication pattern for
   * ghost exchange.
   */
  class Partitioner
  {
  public:
    /**
     * @brief Default constructor.
     */
    Partitioner() = default;

    /**
     * @brief Constructor with locally owned indices and an MPI communicator.
     * Leaves the index set for ghost indices empty.
     *
     * @param local_owned_indices IndexSet of locally owned global indices.
     * @param communicator The MPI communicator to use.
     */
    Partitioner(IndexSet local_owned_indices, MPI_Comm communicator);

    /**
     * @brief Constructor with locally owned and ghost indices and a
     * corresponding MPI communicator.
     *
     * @param local_owned_indices IndexSet of locally owned global indices.
     * @param ghost_indices IndexSet of ghosted global indices.
     * @param communicator The MPI communicator to use.
     */
    Partitioner(IndexSet local_owned_indices,
                IndexSet ghost_indices,
                MPI_Comm communicator);

    /**
     * @brief Checks if a given global index is in the locally owned range.
     *
     * @param global_index The global index to query.
     * @return True if the index is locally owned.
     */
    [[nodiscard]] bool
    in_locally_owned_range(unsigned global_index) const;

    /**
     * @brief Checks if a global index is in the locally relevant range (owned
     * or ghost).
     *
     * @param global_index The global index to query.
     * @return True if the index is either locally owned or ghosted.
     */
    [[nodiscard]] bool
    in_locally_relevant_range(unsigned global_index) const;

    /**
     * @brief Converts a given global index to a local index.
     *
     * @param global_index The global index.
     * @return Corresponding local index.
     */
    [[nodiscard]] unsigned
    global_to_local(unsigned global_index) const;

    /**
     * @brief Converts a given local index to a global index.
     *
     * @param local_index The local index.
     * @return Corresponding global index.
     */
    [[nodiscard]] unsigned
    local_to_global(unsigned local_index) const;

    /**
     * @brief Returns the index set of locally owned indices.
     *
     * @return Reference to the IndexSet of locally owned indices.
     */
    [[nodiscard]] const IndexSet &
    locally_owned_range() const;

    /**
     * @brief Returns the associated MPI communicator.
     *
     * @return MPI communicator used for partitioning.
     */
    [[nodiscard]] MPI_Comm
    get_communicator() const;

    /**
     * @brief Returns the number of locally owned indices.
     *
     * @return Size of the locally owned index set.
     */
    [[nodiscard]] unsigned
    locally_owned_size() const;

    /**
     * @brief Returns the number of ghost indices.
     *
     * @return Size of the ghost index set.
     */
    [[nodiscard]] unsigned
    ghost_size() const;

    /**
     * @brief Checks compatibility with another partitioner.
     *
     * @param partitioner The partitioner to compare with.
     * @return True if both partitioners are compatible, i.e. ghost indices and
     * locally owned indices are the same.
     */
    [[nodiscard]] bool
    is_compatible(const Partitioner &partitioner) const;

    /**
     * @brief Returns the total size of the global index space.
     *
     * @return The total number of global indices.
     */
    [[nodiscard]] unsigned
    get_global_size() const;

    /**
     * @brief Initiates export of data from locally owned to ghosted array using
     * non-blocking communication.
     *
     * @param locally_owned_array The source array of owned data.
     * @param ghost_array The local destination array for ghost data.
     */
    template <std::ranges::contiguous_range ContigView>
    void
    export_to_ghosted_array_start(const ContigView &locally_owned_array,
                                  ContigView       &ghost_array);

    /**
     * @brief Completes the export to the ghosted array after non-blocking MPI
     * communication.
     *
     * @param ghost_array The ghost array to finalize and store the ghost
     * elements in.
     */
    template <std::ranges::contiguous_range ContigView>
    void
    export_to_ghosted_array_finish(ContigView &ghost_array);

  private:
    /**
     * @brief Determines the total number of global indices across all MPI
     * ranks.
     */
    void
    determine_global_size();

    /**
     * @brief Establishes the communication pattern for ghost value exchange.
     */
    void
    determine_communication_pattern();

    /// Total number of global indices.
    unsigned global_size{};

    /// Locally owned global indices.
    mutable IndexSet locally_owned_indices{};

    /// Ghost global indices.
    mutable IndexSet ghost_indices;

    /// List of target MPI ranks and their corresponding indices to which this
    /// process sends data.
    std::vector<std::pair<int, IndexSet>> export_targets_and_indices;

    /// List of source MPI ranks and their corresponding indices from which this
    /// process receives ghost data.
    std::vector<std::pair<int, IndexSet>> import_targets_and_indices;

    /// MPI requests for non-blocking communication used when updating the
    /// ghost values
    std::vector<MPI_Request> request;

    /// The MPI communicator used.
    MPI_Comm communicator{MPI_COMM_SELF};

    /// Rank and size of the MPI communicator.
    int comm_rank{-1}, comm_size{-1};
  };
} // namespace i2phpp


template <std::ranges::contiguous_range ContigView>
void
i2phpp::Partitioner::export_to_ghosted_array_finish(ContigView &ghost_array)
{
  // TODO
}


template <std::ranges::contiguous_range ContigView>
void
i2phpp::Partitioner::export_to_ghosted_array_start(
  const ContigView &locally_owned_array,
  ContigView       &ghost_array)
{
  // TODO
}