#include <gtest/gtest.h>
#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>
#include <i2phpp/util/index_set.hpp>
#include <mpi.h>

#include "test_utils.hpp"

const std::vector<double> test_vector_1({1.2,
                                         3.2,
                                         2.2,
                                         3.4,
                                         1.7,
                                         1.1,
                                         1.7,
                                         9.2,
                                         3.1,
                                         73.2,
                                         3.14,
                                         1.2,
                                         1.0,
                                         3.2,
                                         4.1});
const std::vector<double> test_vector_2(
  {2.4, 3.5, 3.1, 4.3, 1.4, 5.4, 1.2, 4.5, 1.2, 4.3, 5.6, 7.8, 4.6, 9.8, 0.2});

const std::vector<double> expected_sum({3.6,
                                        6.7,
                                        5.3,
                                        7.7,
                                        3.1,
                                        6.5,
                                        2.9,
                                        13.7,
                                        4.3,
                                        77.5,
                                        8.74,
                                        9.0,
                                        5.6,
                                        13.0,
                                        4.3});

const std::vector<double> expected_diff({-1.2,
                                         -0.3,
                                         -0.9,
                                         -0.9,
                                         0.3,
                                         -4.3,
                                         0.5,
                                         4.7,
                                         1.9,
                                         68.9,
                                         -2.46,
                                         -6.6,
                                         -3.6,
                                         -6.6,
                                         3.9});

const std::vector<double> expected_scaled({2.4,
                                           6.4,
                                           4.4,
                                           6.8,
                                           3.4,
                                           2.2,
                                           3.4,
                                           18.4,
                                           6.2,
                                           146.4,
                                           6.28,
                                           2.4,
                                           2.0,
                                           6.4,
                                           8.2});

i2phpp::IndexSet
create_local_index_set(const unsigned global_vector_size,
                       MPI_Comm       communicator = MPI_COMM_WORLD)
{
  int rank, size;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

  unsigned local_size = global_vector_size / size;
  if (rank < size - 1)
    return {rank * local_size, (rank + 1) * local_size};
  else
    return {rank * local_size, global_vector_size};
}


i2phpp::ParallelDistributedVector
create_distributed_vector(const std::vector<double> &values,
                          MPI_Comm communicator = MPI_COMM_WORLD)
{
  int rank, size;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

  i2phpp::ParallelDistributedVector vec(create_local_index_set(values.size(),
                                                               communicator),
                                        communicator);

  unsigned local_size = values.size() / size;

  for (unsigned i = rank * local_size;
       i < (rank < size - 1 ? (rank + 1) * local_size : values.size());
       ++i)
    {
      vec.local_element(i - rank * local_size) = values[i];
    }

  return vec;
}

TEST(DistributedVectorTest, GlobalElement)
{
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  i2phpp::ParallelDistributedVector vec_1(create_local_index_set(size),
                                          MPI_COMM_WORLD);
  for (int i = 0; i < 100; ++i)
    {
      if (rank == size - 1)
        vec_1.local_element(0) = i;
      EXPECT_NEAR(vec_1.gather_global_element(size - 1), i, 1e-10);
    }
}

TEST(DistributedVectorTest, DotProduct)
{
  const i2phpp::ParallelDistributedVector vec_1 =
    create_distributed_vector(test_vector_1);
  const i2phpp::ParallelDistributedVector vec_2 =
    create_distributed_vector(test_vector_2);
  const double vector_product = vec_1.dot_product(vec_2);

  EXPECT_NEAR(vector_product, 469.484, 1e-8);
}

TEST(LaVectorTests, VectorAddition)
{
  i2phpp::ParallelDistributedVector vec_1 =
    create_distributed_vector(test_vector_1);
  const i2phpp::ParallelDistributedVector vec_2 =
    create_distributed_vector(test_vector_2);
  const i2phpp::ParallelDistributedVector sum = vec_1 + vec_2;
  vec_1 += vec_2;

  MPI_Barrier(MPI_COMM_WORLD);
  for (unsigned i = 0; i < vec_1.global_size(); ++i)
    {
      EXPECT_NEAR(vec_1.gather_global_element(i), expected_sum[i], 1e-10);
      EXPECT_NEAR(sum.gather_global_element(i), expected_sum[i], 1e-10);
    }
}

TEST(LaVectorTests, VectorSubtraction)
{
  i2phpp::ParallelDistributedVector vec_1 =
    create_distributed_vector(test_vector_1);
  const i2phpp::ParallelDistributedVector vec_2 =
    create_distributed_vector(test_vector_2);
  const i2phpp::ParallelDistributedVector diff = vec_1 - vec_2;
  vec_1 -= vec_2;

  MPI_Barrier(MPI_COMM_WORLD);
  for (unsigned i = 0; i < vec_1.global_size(); ++i)
    {
      EXPECT_NEAR(vec_1.gather_global_element(i), expected_diff[i], 1e-10);
      EXPECT_NEAR(diff.gather_global_element(i), expected_diff[i], 1e-10);
    }
}

TEST(LaVectorTests, VectorScaling)
{
  i2phpp::ParallelDistributedVector vec =
    create_distributed_vector(test_vector_1);
  constexpr double                        scalar = 2.0;
  const i2phpp::ParallelDistributedVector scaled = vec * scalar;
  vec *= scalar;

  MPI_Barrier(MPI_COMM_WORLD);
  for (unsigned i = 0; i < vec.global_size(); ++i)
    {
      EXPECT_NEAR(vec.gather_global_element(i), expected_scaled[i], 1e-10);
      EXPECT_NEAR(scaled.gather_global_element(i), expected_scaled[i], 1e-10);
    }
}

TEST(LaVectorTests, l2norm)
{
  const i2phpp::ParallelDistributedVector vec =
    create_distributed_vector(test_vector_1);
  const double norm = vec.l2_norm();

  EXPECT_NEAR(norm, 74.343187986526374, 1e-8);
}

MPI_TEST_MAIN;
