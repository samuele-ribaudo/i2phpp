#include <i2phpp/heat_equation/heat_solver.hpp>
#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>
#include <i2phpp/util/time_iterator.hpp>
#include <matplot/core/figure_registry.h>
#include <matplot/matplot.h>

#include <iostream>
#include <memory>
#include <string>

using namespace i2phpp;


template <int dim, typename number>
HeatSolver<dim, number>::HeatSolver(const std::string &input_file)
{
  parse_input(input_file);
}

template <int dim, typename number>
HeatSolver<dim, number>::HeatSolver(
  HeatData<dim, number>    &heat_data,
  TimeIteratorData<number> &time_iterator_data,
  const unsigned int        output_frequency,
  const bool                do_output)
  : heat_data(heat_data)
  , time_iterator_data(time_iterator_data)
  , output_frequency(output_frequency)
  , do_output(do_output)
{}

template <int dim, typename number>
void
HeatSolver<dim, number>::run()
{
  setup_solver();

  // output the initial condition
  write_output();

  // time loop
  while (time_iterator->get_current_time() < time_iterator_data.end_time)
    {
      time_iterator->compute_next_time_increment();

      heat_operator->advance_time_step(
        time_iterator->get_current_time(),
        time_iterator->get_current_time_increment());

      if (time_iterator->get_current_time_step_number() % output_frequency == 0)
        write_output();
    }
}

template <int dim, typename number>
void
HeatSolver<dim, number>::setup_solver()
{
  dof_handler   = std::make_unique<DoFHandler<dim, number>>(heat_data);
  heat_operator = std::make_unique<HeatOperator<dim>>(heat_data, *dof_handler);

  time_iterator = std::make_unique<TimeIterator<number>>(time_iterator_data);
}

template <int dim, typename number>
void
HeatSolver<dim, number>::parse_input(const std::string &input_file)
{
  std::ifstream file_stream(input_file);
  if (!file_stream)
    {
      throw std::runtime_error("Error: Could not open file " + input_file);
    }

  nlohmann::json json_file = nlohmann::json::parse(file_stream);

  json_parse_field(json_file["output"], "frequency", output_frequency);
  json_parse_field(json_file["output"], "enable", do_output);
  json_parse_field(json_file["output"], "layer (3D)", output_layer);

  heat_data.parse_parameters(json_file);
  time_iterator_data.parse_parameters(json_file);
}

template <int dim, typename number>
void
HeatSolver<dim, number>::write_output()
{}

template class i2phpp::HeatSolver<3, double>;
template class i2phpp::HeatSolver<2, double>;
template class i2phpp::HeatSolver<1, double>;