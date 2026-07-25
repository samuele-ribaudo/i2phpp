#include <i2phpp/util/asserts.hpp>
#include <i2phpp/util/partitioner.hpp>
#include <mpi.h>

#include <string>
#include <utility>

i2phpp::Partitioner::Partitioner(i2phpp::IndexSet locally_owned_indices,
                                 i2phpp::IndexSet ghost_indices,
                                 MPI_Comm         communicator)
  : locally_owned_indices(std::move(locally_owned_indices))
  , ghost_indices(std::move(ghost_indices))
  , communicator(communicator)
{
  assert_release(
    this->locally_owned_indices.is_contiguous(),
    std::string(
      "Locally owned index range must be contiguous! Please check the input "
      "parameters!"),
    MPI_COMM_WORLD);

  MPI_Comm_size(this->communicator, &comm_size);
  MPI_Comm_rank(this->communicator, &comm_rank);
  request.reserve(comm_size);
  determine_global_size();
  determine_communication_pattern();
}


i2phpp::Partitioner::Partitioner(i2phpp::IndexSet locally_owned_indices,
                                 MPI_Comm         communicator)
  : locally_owned_indices(std::move(locally_owned_indices))
  , communicator(communicator)
{
  assert_release(
    this->locally_owned_indices.is_contiguous(),
    std::string(
      "Locally owned index range must be contiguous! Please check the input "
      "parameters!"),
    MPI_COMM_WORLD);

  MPI_Comm_size(this->communicator, &comm_size);
  MPI_Comm_rank(this->communicator, &comm_rank);
  determine_global_size();
}


void
i2phpp::Partitioner::determine_global_size()
{
  const unsigned locally_owned_size = locally_owned_indices.size();
  MPI_Allreduce(
    &locally_owned_size, &global_size, 1, MPI_UNSIGNED, MPI_SUM, communicator);
}


void
i2phpp::Partitioner::determine_communication_pattern()
{
  if (comm_size < 2)
    return;

  // Step 1: Send requests to all processes in the communicator
  // TODO
  // gather all ghost indices from all processes into a single array
  const std::vector<unsigned> &my_ghosts      = ghost_indices.get_indices();
  int                          my_ghost_count = my_ghosts.size();

  // Gather the number of ghost indices from all processes to determine the
  // total size of the gathered array and the displacements for each process's
  // data. e.g. if rank 0 has 2 ghosts, rank 1 has 3 ghosts, and rank 2 has 1
  // ghost, the counts would be [2, 3, 1] and the displacements would be [0, 2,
  // 5].
  std::vector<int> ghost_counts(comm_size);
  MPI_Allgather(
    &my_ghost_count, 1, MPI_INT, ghost_counts.data(), 1, MPI_INT, communicator);

  // calculate displacements for the gathered ghost indices e.g. if rank 0 has 2
  // ghosts, rank 1 has 3 ghosts, and rank 2 has 1 ghost, the displacements
  // would be [0, 2, 5] for the gathered array of size 6
  std::vector<int> ghost_displs(comm_size, 0);
  int              total_ghosts = ghost_counts[0];
  for (int i = 1; i < comm_size; i++)
    {
      ghost_displs[i] = ghost_displs[i - 1] + ghost_counts[i - 1];
      total_ghosts += ghost_counts[i];
    }

  // gather all ghost indices from all processes into a single array
  std::vector<unsigned> all_ghosts(total_ghosts);
  MPI_Allgatherv(my_ghosts.data(),
                 my_ghost_count,
                 MPI_UNSIGNED,
                 all_ghosts.data(),
                 ghost_counts.data(),
                 ghost_displs.data(),
                 MPI_UNSIGNED,
                 communicator);

  // Step 2: If any of the locally owned indices match ghost indices of other
  // processes and store them
  // TODO
  for (int r = 0; r < comm_size; r++)
    {
      if (r == comm_rank)
        continue; // Skip our own process

      i2phpp::IndexSet temp;

      // Loop only through the specific chunk of requested ghosts for rank r
      for (int i = 0; i < ghost_counts[r]; i++)
        {
          unsigned index = all_ghosts[ghost_displs[r] + i];

          // Check if this process owns the requested index
          if (locally_owned_indices.is_element(index))
            temp.add_index(index);
        }

      // If we own any of the requested data, add it to our export targets
      if (temp.size() > 0)
        export_targets_and_indices.push_back({r, temp});
    }


  // Step 3: Inform other processes about the any indices which are locally
  // owned and requested by the other processes
  // TODO
  std::vector<MPI_Request> requests;

  // Send the index sets to the respective export targets using MPI_Isend
  for (const auto &[target_rank, index_set] : export_targets_and_indices)
    {
      const auto &indices = index_set.get_indices();
      MPI_Isend(indices.data(),
                indices.size(),
                MPI_UNSIGNED,
                target_rank,
                0, // Use a common tag for all sends
                communicator,
                &request.emplace_back());
    }

  // 2. Post non-blocking receives for incoming import targets
  for (int r = 0; r < comm_size; r++)
    {
      if (r == comm_rank)
        continue; // Skip our own process

      // Determine the number of indices we expect to receive from this source
      int expected_count = ghost_counts[source_rank];

      if (expected_count > 0)
        {
          std::vector<unsigned> received_indices(expected_count);
          MPI_Irecv(received_indices.data(),
                    expected_count,
                    MPI_UNSIGNED,
                    source_rank,
                    0, // Use the same tag as the send
                    communicator,
                    &request.emplace_back());

          // Store the received indices in the import_targets_and_indices vector
          import_targets_and_indices.push_back(
            {source_rank, IndexSet(received_indices)});
        }
    }
  // 3. Wait for all communication requests to complete using MPI_Waitall
  if (!request.empty())
    {
      MPI_Waitall(request.size(), request.data(), MPI_STATUSES_IGNORE);
      request.clear();
    }
}


bool
i2phpp::Partitioner::in_locally_owned_range(unsigned int global_index) const
{
  return locally_owned_indices.is_element(global_index);
}


unsigned
i2phpp::Partitioner::global_to_local(unsigned int global_index) const
{
  if (in_locally_owned_range(global_index))
    {
      return locally_owned_indices.index_within_set(global_index);
    }
  if (in_locally_relevant_range(global_index))
    {
      return locally_owned_indices.size() +
             ghost_indices.index_within_set(global_index);
    }
  else
    {
      assert_release(false,
                     std::string(
                       "The provided global index (" +
                       std::to_string(global_index) +
                       ") is not in the range of locally relevant indices!"),
                     MPI_COMM_WORLD);
    }
  return -1;
}


unsigned
i2phpp::Partitioner::local_to_global(unsigned int local_index) const
{
  return locally_owned_indices.nth_index_in_set(local_index);
}


const i2phpp::IndexSet &
i2phpp::Partitioner::locally_owned_range() const
{
  return locally_owned_indices;
}


MPI_Comm
i2phpp::Partitioner::get_communicator() const
{
  return communicator;
}


unsigned
i2phpp::Partitioner::locally_owned_size() const
{
  return locally_owned_indices.size();
}


unsigned
i2phpp::Partitioner::ghost_size() const
{
  return ghost_indices.size();
}


bool
i2phpp::Partitioner::is_compatible(const i2phpp::Partitioner &partitioner) const
{
  return (std::equal(locally_owned_indices.begin(),
                     locally_owned_indices.end(),
                     partitioner.locally_owned_indices.begin()) &&
          std::equal(ghost_indices.begin(),
                     ghost_indices.end(),
                     partitioner.ghost_indices.begin()));
}

unsigned
i2phpp::Partitioner::get_global_size() const
{
  return global_size;
}

bool
i2phpp::Partitioner::in_locally_relevant_range(unsigned int global_index) const
{
  return in_locally_owned_range(global_index) ||
         ghost_indices.is_element(global_index);
}
