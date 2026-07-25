
#pragma once

#include <mpi.h>

#include <iostream>
#include <ostream>
#include <string>

namespace i2phpp
{
  /**
   * This function checks the given condition and throws a std::runtime_error
   * with the provided message if the condition is false. The check is only
   * performed if the DEBUG macro is defined, allowing for assertions to be
   * included in debug builds without affecting release builds.
   *
   * @param condition The condition to check. If this condition evaluates to
   * false, an exception will be thrown.
   * @param message The error message to include in the exception if the
   * condition is false.
   */
  inline void
  assert_debug(bool               condition,
               const std::string &message,
               MPI_Comm           comm = MPI_COMM_WORLD)
  {
#ifdef DEBUG
    if (!condition)
      {
        std::cerr << message << '\n' << std::flush;
        MPI_Abort(comm, EXIT_FAILURE);
      }
#endif
  }

  /**
   * As above but the check is always performed, regardless of whether the DEBUG
   * macro is defined or not. This can be used for checks that should be
   * enforced in both debug and release builds.
   */
  inline void
  assert_release(bool               condition,
                 const std::string &message,
                 MPI_Comm           comm = MPI_COMM_WORLD)
  {
    if (!condition)
      {
        std::cerr << message << '\n' << std::flush;
        MPI_Abort(comm, EXIT_FAILURE);
      }
  }
} // namespace i2phpp