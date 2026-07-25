/**
 * @brief Holds the relevant data required for solving the heat equation using
 * the finite difference method.
 */

#pragma once

#include <i2phpp/util/asserts.hpp>
#include <i2phpp/util/json_utils.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>

namespace i2phpp
{
  enum class TimeIntegrator
  {
    explicit_euler,
    implicit_euler
  };

  template <int dim, typename number>
  struct HeatData
  {
    struct
    { /// Temperature at the left boundary
      number T_left{};

      /// Temperature at the right boundary
      number T_right{};

      /// Temperature at the front boundary
      number T_front{};

      /// Temperature at the back boundary
      number T_back{};

      /// Temperature at the bottom boundary
      number T_bottom{};

      /// Temperature at the top boundary
      number T_top{};

      /// Boundary conditions (for x, y, z)
      /// The first index describes the spatial direction
      /// while the second index indicates the boundary:
      /// x: 0 - left, 1 - right;
      /// y: 0 - bottom, 1 - top;
      /// z: 0 - front, 1 - back
      std::array<std::array<number, 2>, 3> values{};

    } boundary_conditions;

    /// Initial temperature
    number T_init{};

    /// Domain data
    struct
    {
      /// Length of the domain (x-direction)
      number length{};

      /// Height of the domain (y-direction)
      number height{};

      /// Width of the domain (z-direction)
      number width{};

      /// Domain size in each direction (x, y, z), this is a collection of the
      /// above domain dimension values in an array
      std::array<number, 3> size{};
    } domain;

    /// data struct stroing data from which one can compute the spatially
    /// varying thermal diffusivity
    struct
    {
      /// Base thermal diffusivity value at the origin (x=0, y=0, z=0)
      number base_diffusivity;

      /// Boolean flag indicating whether the thermal diffusivity is constant
      /// across the domain.
      bool constant_thermal_diffusivity;
    } thermal_diffusivity_data;

    /// numerical discretization
    struct
    {
      /// Number of DOFs in the x-direction
      std::size_t nx_dofs{1};

      /// Number of DOFs in the y-direction
      std::size_t ny_dofs{1};

      /// Number of DOFs in the z-direction
      std::size_t nz_dofs{1};

      /// Grid spacing in the x-direction
      number delta_x{};

      /// Grid spacing in the y-direction
      number delta_y{};

      /// Grid spacing in the z-direction
      number delta_z{};

      /// Grid spacing in the x, y, and z directions
      std::array<number, 3> delta_grid{};

      /// Number of DOFs in each direction (x, y, z)
      std::array<std::size_t, 3> n_dofs{};

      /// Total number of global DOFs
      std::size_t n_global_dofs{};
    } grid;

    /// Type of the time integrator. We support explicit and implicit Euler methods.
    TimeIntegrator time_integrator_type = TimeIntegrator::explicit_euler;

    number source_max_energy{0.};

    /// Tolerance used for the stopping criterion of the iterative solver
    number iterative_solver_tolerance{0.01};


    /**
     * @brief Parses parameters from a JSON file and assigns them to class
     * variables.
     *
     * @param json_file The JSON file containing the simulation parameters.
     */
    void
    parse_parameters(const nlohmann::json &json_file)
    {
      json_parse_field(json_file["boundary conditions"],
                       "T left",
                       boundary_conditions.T_left);
      json_parse_field(json_file["boundary conditions"],
                       "T right",
                       boundary_conditions.T_right);
      json_parse_field(json_file["boundary conditions"],
                       "T bottom",
                       boundary_conditions.T_bottom);
      json_parse_field(json_file["boundary conditions"],
                       "T top",
                       boundary_conditions.T_top);
      json_parse_field(json_file["boundary conditions"],
                       "T front",
                       boundary_conditions.T_front);
      json_parse_field(json_file["boundary conditions"],
                       "T back",
                       boundary_conditions.T_back);

      json_parse_field(json_file, "initial temperature", T_init);

      json_parse_field(json_file["domain"], "length", domain.length);
      json_parse_field(json_file["domain"], "height", domain.height);
      json_parse_field(json_file["domain"], "width", domain.width);

      json_parse_field(json_file["discretization"], "nx dofs", grid.nx_dofs);
      json_parse_field(json_file["discretization"], "ny dofs", grid.ny_dofs);
      json_parse_field(json_file["discretization"], "nz dofs", grid.nz_dofs);

      json_parse_field(json_file["heat source"],
                       "max energy",
                       source_max_energy);

      json_parse_field(json_file["thermal diffusivity"],
                       "base diffusivity",
                       thermal_diffusivity_data.base_diffusivity);
      json_parse_field(json_file["thermal diffusivity"],
                       "is constant",
                       thermal_diffusivity_data.constant_thermal_diffusivity);
      json_parse_field(json_file,
                       "iterative_solver_tolerance",
                       iterative_solver_tolerance);


      post();
    }


    /**
     * This function checks the validity of degrees of freedom (DOFs) in each
     * spatial direction and calculates missing parameters (e.g., distance
     * between  DOFs) based on the existing ones. It must be called after the
     * simulation parameters have been read from the input file to properly set
     * up the data structures for the heat simulation.
     */
    void
    post()
    {
      assert_release(grid.nx_dofs > 2, "There must be at least two dofs!");
      assert_release(
        grid.ny_dofs > 0 and grid.nz_dofs > 0,
        "There must be at least one dof in each spatial direction!");

      grid.n_dofs        = {grid.nx_dofs, grid.ny_dofs, grid.nz_dofs};
      grid.delta_x       = domain.length / (grid.nx_dofs - 1);
      grid.delta_y       = domain.height / (grid.ny_dofs - 1);
      grid.delta_z       = domain.width / (grid.nz_dofs - 1);
      grid.n_global_dofs = grid.nx_dofs * grid.ny_dofs * grid.nz_dofs;
      grid.delta_grid    = {grid.delta_x, grid.delta_y, grid.delta_z};
      domain.size        = {domain.length, domain.height, domain.width};

      boundary_conditions.values[0] = {boundary_conditions.T_left,
                                       boundary_conditions.T_right};
      boundary_conditions.values[1] = {boundary_conditions.T_bottom,
                                       boundary_conditions.T_top};
      boundary_conditions.values[2] = {boundary_conditions.T_back,
                                       boundary_conditions.T_front};
    }

    /**
     * @brief Computes the source term at a given location and time.
     *
     * The function calculates the source term based on the spatial position
     * (`location`) and the current time (`current_time`). If the location is
     * within a specific region of the domain, a Gaussian-like decay is applied
     * to the source term. The source term is then scaled by the maximum energy
     * available (`source_max_energy`) and adjusted by the current time to
     * ensure it decreases over time.
     *
     * The spatial region where the source is non-zero is restricted to the area
     * between 25% and 75% of the domain size in each dimension. Outside this
     * region, the source term is zero. The center of the source distribution is
     * at the midpoint of the domain along each dimension, and the decay is
     * Gaussian, centered around this midpoint.
     *
     * @param current_time The current simulation time.
     * @param location The coordinates in the domain where the source term is
     * evaluated.
     *
     * @return The value of the source term at the given location and time.
     */
    number
    source(const number                   current_time,
           const std::array<number, dim> &location) const
    {
      number source = 1.;
      for (unsigned dimension = 0; dimension < dim; ++dimension)
        {
          if (location[dimension] < 0.75 * domain.size[dimension] &&
              location[dimension] > 0.25 * domain.size[dimension])
            {
              source *=
                std::exp(-(location[dimension] - 0.5 * domain.size[dimension]) *
                         (location[dimension] - 0.5 * domain.size[dimension]));
            }
          else
            {
              return 0.;
            }
        }
      source *= 1. / (current_time + 1) * source_max_energy;
      return source;
    }

    /**
     * @brief Returns the thermal diffusivity at the specified location in the
     * domain.
     *
     * If the diffusivity is constant, the function returns the base
     * diffusivity. If the diffusivity is not constant, it calculates and
     * returns the thermal diffusivity at the given location based on a
     * logarithmic gradient along the x-axis. This means that the thermal
     * diffusivity is constant in the y- and z-directions. Additionally, the
     * value at the leftmost end of the x-axis is the base diffusivity. The
     * value at the rightmost end of the x-axis is 0.5 time the base
     * diffusivity.
     *
     * @param location The coordinates at which the diffusivity shall be
     * computed.
     *
     * @return The thermal diffusivity at the specified location.
     */
    inline number
    thermal_diffusivity(const std::array<number, dim> &location) const
    {
      if (thermal_diffusivity_data.constant_thermal_diffusivity)
        return thermal_diffusivity_data.base_diffusivity;
      else
        {
          // just for scaling
          constexpr number exponent = std::log(1e-1);


          return thermal_diffusivity_data.base_diffusivity *
                 std::exp(1. / domain.size[0] * exponent * location[0]);
        }
    }
  };

} // namespace i2phpp