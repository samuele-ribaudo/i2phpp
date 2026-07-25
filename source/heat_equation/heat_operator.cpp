#include <i2phpp/heat_equation/heat_data.hpp>
#include <i2phpp/heat_equation/heat_operator.hpp>
#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>

using namespace i2phpp;

template <int dim>
HeatOperator<dim>::HeatOperator(const HeatData<dim, double>   &heat_data,
                                const DoFHandler<dim, double> &dof_handler)
  : heat_data(heat_data)
  , dof_handler(dof_handler)
  , current_solution(MPI_COMM_WORLD)
  , old_solution(MPI_COMM_WORLD)
{
  // TODO: Properly initialize dof vector (use dof handler function from task 3)
  if (current_solution.get_communicator() != MPI_COMM_NULL)
    {
      apply_boundary_conditions();
      old_solution = current_solution;
    }
}


template <int dim>
void
HeatOperator<dim>::advance_time_step(const double current_time,
                                     const double current_time_increment)
{
  if (current_solution.get_communicator() != MPI_COMM_NULL)
    {
      if (heat_data.time_integrator_type == TimeIntegrator::explicit_euler)
        {
          // TODO: Use correct MPI building blocks

          /// Perform the explicit Euler time step using a matrix-free approach
          current_solution.swap(old_solution);
          const double old_time = current_time - current_time_increment;
          for (unsigned global_index = 0;
               global_index < heat_data.grid.n_global_dofs;
               ++global_index)
            {
              if (!dof_handler.at_boundary(global_index))
                {
                  current_solution.local_element(global_index) =
                    old_solution.local_element(global_index) +
                    current_time_increment *
                      (matrix_free_stencil(global_index, old_solution) +
                       source_term(old_time, global_index));
                }
            }
        }
      else if (heat_data.time_integrator_type == TimeIntegrator::implicit_euler)
        {
          // TODO: Use correct MPI building blocks
          current_solution.swap(old_solution);
          ParallelDistributedVector old_iter = old_solution;
          double                    res  = std::numeric_limits<double>::max();
          unsigned                  iter = 0;
          while (res > heat_data.iterative_solver_tolerance && iter < 1000)
            {
              for (unsigned global_index = 0;
                   global_index < heat_data.grid.n_global_dofs;
                   ++global_index)
                {
                  if (!dof_handler.at_boundary(global_index))
                    {
                      current_solution.local_element(global_index) =
                        jacobi_update(global_index,
                                      current_time,
                                      current_time_increment,
                                      old_iter,
                                      old_solution);
                    }
                }
              res = 0.;
              for (unsigned global_index = 0;
                   global_index < heat_data.grid.n_global_dofs;
                   ++global_index)
                {
                  if (!dof_handler.at_boundary(global_index))
                    {
                      double loc_res =
                        old_solution.local_element(global_index) +
                        current_time_increment *
                          (matrix_free_stencil(global_index, current_solution) +
                           source_term(current_time, global_index)) -
                        current_solution.local_element(global_index);
                      res += loc_res * loc_res;
                    }
                }
              MPI_Allreduce(&res,
                            &res,
                            1,
                            MPI_DOUBLE,
                            MPI_SUM,
                            current_solution.get_communicator());
              res      = std::sqrt(res) / current_solution.l2_norm();
              old_iter = current_solution;
              ++iter;
            }
        }
      else
        assert_release(false,
                       "The provided solver type is not supported!",
                       MPI_COMM_WORLD);
    }
}

template <int dim>
const ParallelDistributedVector &
HeatOperator<dim>::get_current_solution() const
{
  return current_solution;
}

template <int dim>
ParallelDistributedVector &
HeatOperator<dim>::get_current_solution()
{
  return current_solution;
}

template <int dim>
void
HeatOperator<dim>::apply_boundary_conditions()
{
  // TODO: Use correct MPI building blocks
  if (current_solution.get_communicator() != MPI_COMM_NULL)
    {
      for (unsigned global_index = 0;
           global_index < current_solution.locally_owned_size();
           ++global_index)
        {
          if (dof_handler.at_boundary(global_index))
            {
              for (unsigned int dimension = 0; dimension < dim; ++dimension)
                if (dof_handler.get_cartesian_index(global_index)[dimension] ==
                    0)
                  {
                    current_solution.local_element(global_index) =
                      heat_data.boundary_conditions.values[dimension][0];
                  }
                else if (dof_handler.get_cartesian_index(
                           global_index)[dimension] ==
                         heat_data.grid.n_dofs[dimension] - 1)
                  current_solution.local_element(global_index) =
                    heat_data.boundary_conditions.values[dimension][1];
            }
        }
    }
}

template <int dim>
double
HeatOperator<dim>::source_term(const double      current_time,
                               const std::size_t index) const
{
  std::array<double, dim> location = dof_handler.get_physical_location(index);
  return heat_data.source(current_time, location);
}

template <int dim>
double
HeatOperator<dim>::matrix_free_stencil(
  const std::size_t                global_index,
  const ParallelDistributedVector &solution) const
{
  // TODO: Use correct MPI building blocks
  std::array<std::size_t, dim> offset{};
  offset[0] = 1;
  for (std::size_t i = 1; i < dim; ++i)
    {
      offset[i] = offset[i - 1] * heat_data.grid.n_dofs[i - 1];
    }

  auto   local_indices = dof_handler.get_cartesian_index(global_index);
  double result        = 0;
  for (unsigned int dimension = 0; dimension < dim; ++dimension)
    {
      auto delta_grid_squared_inverse =
        1. / (heat_data.grid.delta_grid[dimension] *
              heat_data.grid.delta_grid[dimension]);
      result += heat_data.thermal_diffusivity(
                  dof_handler.get_physical_location(local_indices)) *
                delta_grid_squared_inverse *
                (solution.local_element(global_index - offset[dimension]) -
                 2. * solution.local_element(global_index) +
                 solution.local_element(global_index + offset[dimension]));

      result +=
        0.25 * delta_grid_squared_inverse *
        (heat_data.thermal_diffusivity(dof_handler.get_physical_location(
           global_index + offset[dimension])) -
         heat_data.thermal_diffusivity(dof_handler.get_physical_location(
           global_index - offset[dimension]))) *
        (solution.local_element(global_index + offset[dimension]) -
         solution.local_element(global_index - offset[dimension]));
    }
  return result;
}

template <int dim>
double
HeatOperator<dim>::jacobi_update(
  const std::size_t                global_index,
  const double                     time,
  const double                     current_time_increment,
  const ParallelDistributedVector &old_iter_vector,
  const ParallelDistributedVector &rhs) const
{
  // TODO: Use correct MPI building blocks
  std::array<std::size_t, dim> offset{};
  offset[0] = 1;
  for (std::size_t i = 1; i < dim; ++i)
    {
      offset[i] = offset[i - 1] * heat_data.grid.n_dofs[i - 1];
    }

  auto   cartesian_indices = dof_handler.get_cartesian_index(global_index);
  double result            = rhs.local_element(global_index) +
                  current_time_increment * source_term(time, global_index);
  double multiplicator = 0;
  for (unsigned int dimension = 0; dimension < dim; ++dimension)
    {
      auto delta_grid_squared_inverse =
        1. / (heat_data.grid.delta_grid[dimension] *
              heat_data.grid.delta_grid[dimension]);

      result +=
        current_time_increment *
        heat_data.thermal_diffusivity(
          dof_handler.get_physical_location(cartesian_indices)) *
        delta_grid_squared_inverse *
        (old_iter_vector.local_element(global_index - offset[dimension]) +
         old_iter_vector.local_element(global_index + offset[dimension]));

      result +=
        current_time_increment * 0.25 * delta_grid_squared_inverse *
        (heat_data.thermal_diffusivity(dof_handler.get_physical_location(
           global_index + offset[dimension])) -
         heat_data.thermal_diffusivity(dof_handler.get_physical_location(
           global_index - offset[dimension]))) *
        (old_iter_vector.local_element(global_index + offset[dimension]) -
         old_iter_vector.local_element(global_index - offset[dimension]));
      multiplicator += delta_grid_squared_inverse;
    }
  multiplicator *= heat_data.thermal_diffusivity(
                     dof_handler.get_physical_location(cartesian_indices)) *
                   2 * current_time_increment;
  return result / (multiplicator + 1);
}

template class i2phpp::HeatOperator<3>;
template class i2phpp::HeatOperator<2>;
template class i2phpp::HeatOperator<1>;