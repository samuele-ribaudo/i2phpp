#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>
#include <i2phpp/util/asserts.hpp>
#include <i2phpp/util/index_set.hpp>
#include <i2phpp/util/partitioner.hpp>
#include <mpi.h>

#include <cmath>
#include <span>
#include <string>

using namespace i2phpp;

ParallelDistributedVector::ParallelDistributedVector(MPI_Comm communicator)
  : communicator(communicator)
{}


ParallelDistributedVector::ParallelDistributedVector(
  const Partitioner &partitioner,
  double             initial_value)
  : partitioner(partitioner)
  , communicator(partitioner.get_communicator())
  , local_elements(partitioner.locally_owned_size() + partitioner.ghost_size(),
                   initial_value)
{}


ParallelDistributedVector::ParallelDistributedVector(
  const IndexSet &local_range,
  MPI_Comm        communicator,
  double          initial_value)
  : partitioner(local_range, communicator)
  , communicator(communicator)
  , local_elements(local_range.size(), initial_value)
{}


i2phpp::ParallelDistributedVector::ParallelDistributedVector(
  const i2phpp::IndexSet &local_range,
  const i2phpp::IndexSet &ghost_indices,
  MPI_Comm                communicator,
  double                  initial_value)
  : partitioner(local_range, ghost_indices, communicator)
  , communicator(communicator)
  , local_elements(local_range.size() + ghost_indices.size(), initial_value)
{}


void
i2phpp::ParallelDistributedVector::reinit(const IndexSet &locally_owned_indices,
                                          const IndexSet &ghost_indices,
                                          const double    initial_value,
                                          MPI_Comm        comm)
{
  communicator = comm;
  partitioner = Partitioner(locally_owned_indices, ghost_indices, communicator);
  local_elements.clear();
  local_elements.resize(locally_owned_size() + partitioner.ghost_size(),
                        initial_value);
}


double &
i2phpp::ParallelDistributedVector::local_element(const unsigned local_index)
{
  return local_elements[local_index];
}


const double &
i2phpp::ParallelDistributedVector::local_element(
  const unsigned local_index) const
{
  return local_elements[local_index];
}


double &
i2phpp::ParallelDistributedVector::global_element(unsigned int global_index)
{
  return local_elements[partitioner.global_to_local(global_index)];
}


const double &
i2phpp::ParallelDistributedVector::global_element(
  unsigned int global_index) const
{
  return local_elements[partitioner.global_to_local(global_index)];
}


double
i2phpp::ParallelDistributedVector::gather_global_element(
  const unsigned global_index) const
{
  assert_release(global_index < partitioner.get_global_size(),
                 std::string(
                   "The global index you have passed to the function is great "
                   "than the total (global) number of elements in the vector."),
                 communicator);

  double element;
  int    size, rank;
  MPI_Comm_size(communicator, &size);
  MPI_Comm_rank(communicator, &rank);
  if (partitioner.in_locally_owned_range(global_index))
    {
      element = local_elements[partitioner.global_to_local(global_index)];
      for (int receiver = 0; receiver < size; ++receiver)
        {
          if (receiver != rank)
            MPI_Send(&element,
                     1,
                     MPI_DOUBLE,
                     receiver,
                     static_cast<int>(global_index),
                     communicator);
        }
    }
  else
    {
      MPI_Recv(&element,
               1,
               MPI_DOUBLE,
               MPI_ANY_SOURCE,
               static_cast<int>(global_index),
               communicator,
               MPI_STATUS_IGNORE);
    }
  MPI_Barrier(communicator);
  return element;
}


unsigned
i2phpp::ParallelDistributedVector::locally_owned_size() const
{
  return partitioner.locally_owned_size();
}


unsigned
i2phpp::ParallelDistributedVector::ghost_size() const
{
  return partitioner.ghost_size();
}


double
i2phpp::ParallelDistributedVector::dot_product(
  const i2phpp::ParallelDistributedVector &other) const
{
  // TODO
  double local_sum = 0.0, global_sum;
  assert_compatibility(other);
  for (auto i = 0; i < locally_owned_size(); i++)
    local_sum += local_elements[i] * other.local_elements[i];

  // use allreduce such that if all the processes need to use the result, they
  // can do so without having to send it to each other
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator);

  return global_sum;
}


double
i2phpp::ParallelDistributedVector::l2_norm() const
{
  // TODO
  double local_sum = 0.0, global_sum;
  for (auto i = 0; i < locally_owned_size(); i++)
    local_sum += local_elements[i] * local_elements[i];

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator);
  return std::sqrt(global_sum);
}


void
i2phpp::ParallelDistributedVector::operator*=(const double num)
{
  // TODO
  for (auto i = 0; i < locally_owned_size() + ghost_size();
       i++) // include ghost elements as well
    local_elements[i] *= num;
}


i2phpp::ParallelDistributedVector
i2phpp::ParallelDistributedVector::operator*(const double num) const
{
  // TODO
  ParallelDistributedVector result(partitioner);
  for (auto i = 0; i < locally_owned_size() + ghost_size(); i++)
    result.local_elements[i] = local_elements[i] * num;
  return result;
}


void
i2phpp::ParallelDistributedVector::operator+=(
  const ParallelDistributedVector &vec)
{
  // TODO
  assert_compatibility(vec);
  for (auto i = 0; i < locally_owned_size() + ghost_size(); i++)
    local_elements[i] += vec.local_elements[i];
}


i2phpp::ParallelDistributedVector
i2phpp::ParallelDistributedVector::operator+(
  const ParallelDistributedVector &vec) const
{
  // TODO
  assert_compatibility(vec);
  ParallelDistributedVector result(partitioner);
  for (auto i = 0; i < locally_owned_size() + ghost_size(); i++)
    result.local_elements[i] = local_elements[i] + vec.local_elements[i];
  return result;
}


void
i2phpp::ParallelDistributedVector::operator-=(
  const ParallelDistributedVector &vec)
{
  // TODO
  assert_compatibility(vec);
  for (auto i = 0; i < locally_owned_size() + ghost_size(); i++)
    local_elements[i] -= vec.local_elements[i];
}

i2phpp::ParallelDistributedVector
i2phpp::ParallelDistributedVector::operator-(
  const ParallelDistributedVector &vec) const
{
  // TODO
  assert_compatibility(vec);
  ParallelDistributedVector result(partitioner);
  for (auto i = 0; i < locally_owned_size() + ghost_size(); i++)
    result.local_elements[i] = local_elements[i] - vec.local_elements[i];
  return result;
}

void
i2phpp::ParallelDistributedVector::update_ghost_values()
{
  std::span<double> locally_owned_elements(local_elements.data(),
                                           locally_owned_size());
  std::span<double> ghost_elements(local_elements.data() + locally_owned_size(),
                                   partitioner.ghost_size());

  partitioner.export_to_ghosted_array_start(locally_owned_elements,
                                            ghost_elements);
  partitioner.export_to_ghosted_array_finish(ghost_elements);
}

bool
i2phpp::ParallelDistributedVector::is_compatible(
  const i2phpp::ParallelDistributedVector &other) const
{
  return partitioner.is_compatible(other.partitioner);
}

unsigned
i2phpp::ParallelDistributedVector::global_size() const
{
  return partitioner.get_global_size();
}

void
i2phpp::ParallelDistributedVector::swap(
  i2phpp::ParallelDistributedVector &other)
{
  local_elements.swap(other.local_elements);
}


const i2phpp::Partitioner &
i2phpp::ParallelDistributedVector::get_partitioner() const
{
  return partitioner;
}


MPI_Comm
i2phpp::ParallelDistributedVector::get_communicator() const
{
  return communicator;
}


void
i2phpp::ParallelDistributedVector::assert_compatibility(
  const i2phpp::ParallelDistributedVector &other) const
{
  assert_release(is_compatible(other),
                 std::string(
                   "The parallel layout of the two distributed vectors you are "
                   "working on is not compatible!"),
                 communicator);
}