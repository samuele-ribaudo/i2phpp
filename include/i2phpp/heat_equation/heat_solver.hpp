
#pragma once

#include <i2phpp/heat_equation/heat_data.hpp>
#include <i2phpp/heat_equation/heat_operator.hpp>
#include <i2phpp/util/time_iterator.hpp>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>

namespace i2phpp
{
  template <int dim, typename number>
  class HeatSolver
  {
  public:
    /**
     * Constructor that reads the required simulation data from the provided
     * JSON input file.
     *
     * @param input_file The path to the input JSON file containing the
     * simulation data.
     */
    explicit HeatSolver(const std::string &input_file);

    /**
     * Constructor that takes externally provided data instead of reading it
     * from an input file. This constructor is primarily implemented for
     * testing purposes.
     *
     * @param heat_data Externally defined heat data to be used in the solver.
     * @param time_iterator_data Externally defined time iterator data.
     * @param output_frequency Frequency at which the output is generated.
     * @param do_output Flag to enable or disable output generation.
     */
    HeatSolver(HeatData<dim, number>    &heat_data,
               TimeIteratorData<number> &time_iterator_data,
               unsigned int              output_frequency = 100,
               bool                      do_output        = false);

    /**
     * Run the simulation. This function does all necessary calls to prepare the
     * simulation and executes the time loop.
     */
    void
    run();

    /**
     * Retrieves the current solution. This function is primarily used for
     * testing purposes.
     *
     * @return A constant reference to the current solution stored in the heat
     * operator.
     */
    const ParallelDistributedVector &
    get_solution() const
    {
      return heat_operator->get_current_solution();
    }

  private:
    /**
     * Initializes all necessary data structures for the solver. This includes
     * creating objects for the DoF handler, heat operator, and time iterator by
     * calling the appropriate constructors.
     */
    void
    setup_solver();

    /**
     * Parses the input data from the specified JSON file.
     *
     * @param input_file The path to the input JSON file containing solver data.
     */
    void
    parse_input(const std::string &input_file);

    /**
     * Generates and saves a plot of the current solution to a PNG file.
     *
     * For the 1D case, a simple line plot is generated. In 2D, a surface plot
     * is produced. In 3D, the plot is currently restricted to a single layer in
     * the z-direction, visualized as a surface plot. The specific layer to plot
     * is determined by the parameter @param output_layer which can be
     * specified in the input file.
     */
    void
    write_output();

    std::unique_ptr<HeatOperator<dim>> heat_operator;

    std::unique_ptr<TimeIterator<number>> time_iterator;

    HeatData<dim, number>    heat_data{};
    TimeIteratorData<number> time_iterator_data{};

    std::unique_ptr<DoFHandler<dim, number>> dof_handler;

    /// output frequency
    unsigned int output_frequency = 1;

    /// helper for indexing output files
    unsigned int output_idx = 0;

    /// if set, output is written into png files
    bool do_output = true;

    /// output layer to output a 2d plot for the three dimensional solution in
    /// the 3d case
    std::size_t output_layer = 1;
  };
} // namespace i2phpp