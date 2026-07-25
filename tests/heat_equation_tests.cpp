/**
 * @brief Tests for the heat equation solver.
 */

#include <gtest/gtest.h>
#include <i2phpp/heat_equation/heat_data.hpp>
#include <i2phpp/heat_equation/heat_solver.hpp>
#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>
#include <i2phpp/util/time_iterator.hpp>

#include "test_utils.hpp"

TEST(HeatEquationTest, OneDimensionalHeatEquationExplicit)
{
  constexpr unsigned int        dim = 1;
  i2phpp::HeatData<dim, double> heat_data{};
  heat_data.thermal_diffusivity_data.base_diffusivity             = 1e-2;
  heat_data.thermal_diffusivity_data.constant_thermal_diffusivity = true;
  heat_data.boundary_conditions.T_left                            = 273;
  heat_data.boundary_conditions.T_right                           = 303;
  heat_data.T_init                                                = 410;
  heat_data.domain.length                                         = 1.0;
  heat_data.grid.nx_dofs                                          = 100;
  heat_data.time_integrator_type = i2phpp::TimeIntegrator::explicit_euler;
  heat_data.post();

  i2phpp::TimeIteratorData<double> time_iterator_data;
  time_iterator_data.end_time               = 1;
  time_iterator_data.default_time_increment = 0.001;

  i2phpp::HeatSolver<dim, double> solver(heat_data, time_iterator_data);
  solver.run();

  if (solver.get_solution().get_communicator() != MPI_COMM_NULL)
    {
      EXPECT_NEAR(solver.get_solution().gather_global_element(4),
                  303.8051065008359,
                  1e-8);
      EXPECT_NEAR(solver.get_solution().gather_global_element(91),
                  349.24509340826182,
                  1e-8);
    }
}

TEST(HeatEquationTest, TwoDimensionalHeatEquationExplicit)
{
  constexpr unsigned int        dim = 2;
  i2phpp::HeatData<dim, double> heat_data;
  heat_data.thermal_diffusivity_data.base_diffusivity             = 1e-2;
  heat_data.thermal_diffusivity_data.constant_thermal_diffusivity = true;
  heat_data.boundary_conditions.T_left                            = 273;
  heat_data.boundary_conditions.T_right                           = 303;
  heat_data.boundary_conditions.T_bottom                          = 293;
  heat_data.boundary_conditions.T_top                             = 323;
  heat_data.T_init                                                = 410;
  heat_data.domain.length                                         = 1.0;
  heat_data.domain.height                                         = 1.0;
  heat_data.grid.nx_dofs                                          = 20;
  heat_data.grid.ny_dofs                                          = 20;
  heat_data.time_integrator_type = i2phpp::TimeIntegrator::explicit_euler;
  heat_data.post();

  i2phpp::TimeIteratorData<double> time_iterator_data;
  time_iterator_data.end_time               = 0.5;
  time_iterator_data.default_time_increment = 0.001;


  i2phpp::HeatSolver<dim, double> solver(heat_data, time_iterator_data);
  solver.run();

  if (solver.get_solution().get_communicator() != MPI_COMM_NULL)
    {
      EXPECT_NEAR(solver.get_solution().gather_global_element(25),
                  339.57498125201209,
                  1e-8);
      EXPECT_NEAR(solver.get_solution().gather_global_element(297),
                  377.77730647108007,
                  1e-8);
    }
}

TEST(HeatEquationTest, TwoDimensionalHeatEquationWithSourceExplicit)
{
  constexpr unsigned int        dim = 2;
  i2phpp::HeatData<dim, double> heat_data;
  heat_data.thermal_diffusivity_data.base_diffusivity             = 1e-2;
  heat_data.thermal_diffusivity_data.constant_thermal_diffusivity = true;
  heat_data.boundary_conditions.T_left                            = 273;
  heat_data.boundary_conditions.T_right                           = 303;
  heat_data.boundary_conditions.T_bottom                          = 293;
  heat_data.boundary_conditions.T_top                             = 323;
  heat_data.T_init                                                = 410;
  heat_data.domain.length                                         = 1.0;
  heat_data.domain.height                                         = 1.0;
  heat_data.grid.nx_dofs                                          = 20;
  heat_data.grid.ny_dofs                                          = 20;
  heat_data.source_max_energy                                     = 50.;
  heat_data.time_integrator_type = i2phpp::TimeIntegrator::explicit_euler;
  heat_data.post();

  i2phpp::TimeIteratorData<double> time_iterator_data;
  time_iterator_data.end_time               = 0.5;
  time_iterator_data.default_time_increment = 0.001;


  i2phpp::HeatSolver<dim, double> solver(heat_data, time_iterator_data);
  solver.run();

  if (solver.get_solution().get_communicator() != MPI_COMM_NULL)
    {
      EXPECT_NEAR(solver.get_solution().gather_global_element(25),
                  339.69656626201453,
                  1e-8);
      EXPECT_NEAR(solver.get_solution().gather_global_element(297),
                  378.23044873661564,
                  1e-8);
    }
}

TEST(HeatEquationTest, ThreeDimensionalHeatEquationExplicit)
{
  constexpr unsigned int        dim = 3;
  i2phpp::HeatData<dim, double> heat_data;
  heat_data.thermal_diffusivity_data.base_diffusivity             = 1e-2;
  heat_data.thermal_diffusivity_data.constant_thermal_diffusivity = true;
  heat_data.boundary_conditions.T_left                            = 273;
  heat_data.boundary_conditions.T_right                           = 303;
  heat_data.boundary_conditions.T_bottom                          = 293;
  heat_data.boundary_conditions.T_top                             = 323;
  heat_data.boundary_conditions.T_front                           = 300;
  heat_data.boundary_conditions.T_back                            = 330;
  heat_data.T_init                                                = 410;
  heat_data.domain.length                                         = 1.0;
  heat_data.domain.height                                         = 1.0;
  heat_data.domain.width                                          = 1.0;
  heat_data.grid.nx_dofs                                          = 10;
  heat_data.grid.ny_dofs                                          = 10;
  heat_data.grid.nz_dofs                                          = 10;
  heat_data.time_integrator_type = i2phpp::TimeIntegrator::explicit_euler;
  heat_data.post();

  i2phpp::TimeIteratorData<double> time_iterator_data;
  time_iterator_data.end_time               = 0.05;
  time_iterator_data.default_time_increment = 0.003;


  i2phpp::HeatSolver<dim, double> solver(heat_data, time_iterator_data);
  solver.run();

  if (solver.get_solution().get_communicator() != MPI_COMM_NULL)
    {
      EXPECT_NEAR(solver.get_solution().gather_global_element(321),
                  404.57314631926585,
                  1e-8);
      EXPECT_NEAR(solver.get_solution().gather_global_element(787),
                  406.45237724402813,
                  1e-8);
    }
}


TEST(HeatEquationTest, ThreeDimensionalHeatEquationExplicitSourceThermal)
{
  constexpr unsigned int        dim = 3;
  i2phpp::HeatData<dim, double> heat_data;
  heat_data.thermal_diffusivity_data.base_diffusivity = 1e-2;
  heat_data.boundary_conditions.T_left                = 273;
  heat_data.boundary_conditions.T_right               = 303;
  heat_data.boundary_conditions.T_bottom              = 293;
  heat_data.boundary_conditions.T_top                 = 323;
  heat_data.boundary_conditions.T_front               = 300;
  heat_data.boundary_conditions.T_back                = 330;
  heat_data.T_init                                    = 410;
  heat_data.domain.length                             = 1.0;
  heat_data.domain.height                             = 1.0;
  heat_data.domain.width                              = 1.0;
  heat_data.grid.nx_dofs                              = 10;
  heat_data.grid.ny_dofs                              = 10;
  heat_data.grid.nz_dofs                              = 10;
  heat_data.source_max_energy                         = 50.;
  heat_data.time_integrator_type = i2phpp::TimeIntegrator::explicit_euler;
  heat_data.thermal_diffusivity_data.constant_thermal_diffusivity = false;
  heat_data.post();

  i2phpp::TimeIteratorData<double> time_iterator_data;
  time_iterator_data.end_time               = 0.05;
  time_iterator_data.default_time_increment = 0.003;


  i2phpp::HeatSolver<dim, double> solver(heat_data, time_iterator_data);
  solver.run();

  if (solver.get_solution().get_communicator() != MPI_COMM_NULL)
    {
      EXPECT_NEAR(solver.get_solution().gather_global_element(321),
                  405.23792727330255,
                  1e-8);
      EXPECT_NEAR(solver.get_solution().gather_global_element(787),
                  409.41205243604617,
                  1e-8);
    }
}

TEST(HeatEquationTest, ThreeDimensionalHeatEquationImplicitSourceThermal)
{
  constexpr unsigned int        dim = 3;
  i2phpp::HeatData<dim, double> heat_data;
  heat_data.thermal_diffusivity_data.base_diffusivity = 1e-2;
  heat_data.boundary_conditions.T_left                = 273;
  heat_data.boundary_conditions.T_right               = 303;
  heat_data.boundary_conditions.T_bottom              = 293;
  heat_data.boundary_conditions.T_top                 = 323;
  heat_data.boundary_conditions.T_front               = 300;
  heat_data.boundary_conditions.T_back                = 330;
  heat_data.T_init                                    = 410;
  heat_data.domain.length                             = 1.0;
  heat_data.domain.height                             = 1.0;
  heat_data.domain.width                              = 1.0;
  heat_data.grid.nx_dofs                              = 10;
  heat_data.grid.ny_dofs                              = 10;
  heat_data.grid.nz_dofs                              = 10;
  heat_data.source_max_energy                         = 50.;
  heat_data.time_integrator_type       = i2phpp::TimeIntegrator::implicit_euler;
  heat_data.iterative_solver_tolerance = 0.02;
  heat_data.thermal_diffusivity_data.constant_thermal_diffusivity = false;
  heat_data.post();

  i2phpp::TimeIteratorData<double> time_iterator_data;
  time_iterator_data.end_time               = 0.2;
  time_iterator_data.default_time_increment = 0.01;


  i2phpp::HeatSolver<dim, double> solver(heat_data, time_iterator_data);
  solver.run();

  if (solver.get_solution().get_communicator() != MPI_COMM_NULL)
    {
      EXPECT_NEAR(solver.get_solution().gather_global_element(321),
                  392.64803398130647,
                  1e-8);
      EXPECT_NEAR(solver.get_solution().gather_global_element(787),
                  407.66625400382185,
                  1e-8);
    }
}

MPI_TEST_MAIN;
