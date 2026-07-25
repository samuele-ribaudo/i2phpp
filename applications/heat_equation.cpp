/**
 * @file heat_equation.cpp
 * @brief Entry point for the heat equation solver.
 */
#include <i2phpp/heat_equation/heat_solver.hpp>

#include <exception>
#include <string>
#include <iostream>
#include <fstream>

using namespace i2phpp;

// change if you want to try different types like float
typedef double base_type;

int
main(int argc, char *argv[])
{
  // check for the right number of program arguments
  if (argc < 2)
    {
      throw std::runtime_error("You must provide an input file.");
    }
  else if (argc > 2)
    {
      throw std::runtime_error(
        "To many input arguments provided. You should only pass one argument - "
        "the name of the input file.");
    }

  // get the input file name
  std::string input_file = argv[argc - 1];

  // start the actual solver
  try
    {
      unsigned int dim;
      {
        std::ifstream file_stream(input_file);
        if (!file_stream)
          {
            throw std::runtime_error("Error: Could not open file " +
                                     input_file);
          }

        nlohmann::json json_file = nlohmann::json::parse(file_stream);

        json_parse_field(json_file, "dimension", dim);
      }
      switch (dim)
        {
          case 1:
            {
              i2phpp::HeatSolver<1, base_type> solver(input_file);
              solver.run();
              break;
            }
          case 2:
            {
              i2phpp::HeatSolver<2, base_type> solver(input_file);
              solver.run();
              break;
            }
          case 3:
            {
              i2phpp::HeatSolver<3, base_type> solver(input_file);
              solver.run();
              break;
            }
          default:
            throw std::invalid_argument("The passed dimension \'" +
                                        std::to_string(dim) +
                                        "\' is not supported!");
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << "\n\n----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
    }
  catch (...)
    {
      std::cerr << "\n\n----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
    }

  return 0;
}
