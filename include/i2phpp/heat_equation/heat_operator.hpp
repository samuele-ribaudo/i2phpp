
#pragma once

#include <i2phpp/heat_equation/heat_data.hpp>
#include <i2phpp/heat_equation/heat_operator.hpp>
#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>
#include <i2phpp/util/dof_handler.hpp>

namespace i2phpp
{
  template <int dim>
  class HeatOperator
  {
  public:
    /**
     * Constructor that computes the system matrix and initializes the solution
     * vector with the initial conditions. The initial solution is then adjusted
     * to satisfy the boundary conditions.
     *
     * @param heat_data Data required to solve the heat equation.
     * @param dof_handler Helper funcctions for DoF indexing.
     */
    HeatOperator(const HeatData<dim, double>   &heat_data,
                 const DoFHandler<dim, double> &dof_handler);

    /**
     * Advances the solution by one time step, from T^(n) to T^(n+1), using the
     * explicit Euler scheme. The solution at the new time step is computed and
     * stored internally, replacing the previous result.
     *
     * @param current_time Current simulation time at t^(n+1)
     * @param current_time_increment The time step size, dt = t^(n+1) - t^(n).
     */
    void
    advance_time_step(double current_time, double current_time_increment);

    /**
     * @brief Returns a constant reference to the current solution stored in
     * the solver.
     */
    const ParallelDistributedVector &
    get_current_solution() const;

    /**
     * @brief Returns a modifiable reference to the current solution stored in
     * the solver.
     */
    ParallelDistributedVector &
    get_current_solution();

  private:
    /**
     * Applies the boundary conditions to the solution vector. The boundary
     * condition values are retrieved from the HeatData struct, which is stored
     * as a reference by the class.
     */
    void
    apply_boundary_conditions();

    /**
     * @brief Computes the thermal source term at a given time for the point
     * corresponding to a specific array index.
     *
     * @param current_time The current simulation time.
     * @param index The global index of the point at which the source term is
     * evaluated.
     *
     * @return The computed source term value at the given index and time.
     */
    double
    source_term(const double current_time, const std::size_t index) const;

    /**
     * @brief Computes the finite difference stencil for the heat equation at a
     * given global index.
     *
     * This function evaluates the discrete Laplace operator while accounting
     * for spatially varying thermal diffusivity. The computation is performed
     * in a matrix-free manner, operating directly on the solution vector
     * without assembling a global matrix.
     *
     * @param global_index The global degree of freedom (DoF) index at which 
     * the stencil is evaluated.
     * @param solution The solution vector on which the stencil is applied.
     *
     * @return The computed stencil value at the specified index.
     */
    double
    matrix_free_stencil(const std::size_t                global_index,
                        const ParallelDistributedVector &solution) const;

    /**
     * This function computes the Jacobi update for a given global index. For
     * theoretical background, refer to equation (8) in the second worksheet.
     *
     * @param global_index The global degree of freedom (DoF) index at which the
     * Jacobi update is evaluated.
     * @param time The current simulation time at t^(n+1).
     * @param current_time_increment The time step size, dt = t^(n+1) - t^(n).
     * @param old_iter_vector The solution vector from the previous iteration.
     * @param rhs The right-hand side vector for the Jacobi update.
     */
    double
    jacobi_update(const std::size_t                global_index,
                  const double                     time,
                  const double                     current_time_increment,
                  const ParallelDistributedVector &old_iter_vector,
                  const ParallelDistributedVector &rhs) const;



    /// Heat data containing all necessary parameters and properties for solving
    /// the heat equation.
    const HeatData<dim, double> heat_data;

    /// Vector holding the current solution (temperature field) at the current
    /// time t^(n+1)AS.
    ParallelDistributedVector current_solution;

    /// Vector holding the old solution (temperature field) at the previous time
    /// t^(n).
    ParallelDistributedVector old_solution;

    /// Reference to the DoF handler that manages degree-of-freedom indexing.
    const DoFHandler<dim, double> &dof_handler;
  };
} // namespace i2phpp
