#include <gtest/gtest.h>
#include <mpi.h>

#define MPI_TEST_MAIN                                                 \
  int main(int argc, char **argv)                                     \
  {                                                                   \
    testing::InitGoogleTest(&argc, argv);                             \
    MPI_Init(nullptr, nullptr);                                       \
    ::testing::TestEventListeners &listeners =                        \
      ::testing::UnitTest::GetInstance()->listeners();                \
    int rank, size;                                                   \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                             \
    MPI_Comm_size(MPI_COMM_WORLD, &size);                             \
    if (rank != 0)                                                    \
      {                                                               \
        delete listeners.Release(listeners.default_result_printer()); \
      }                                                               \
    int result = RUN_ALL_TESTS();                                     \
    MPI_Finalize();                                                   \
    return result;                                                    \
  }
  