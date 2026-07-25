/**
 * @brief Manages the degrees of freedom in the domain.
 *
 * For the finite difference implementation, this class provides convenient
 * operations for both array and cartesian indexing.
 */

#pragma once

#include <i2phpp/heat_equation/heat_data.hpp>
#include <i2phpp/linear_algebra/parallel_distributed_vector.hpp>

#include <array>
#include <cstddef>

namespace i2phpp
{
  template <int dim, typename number>
  class DoFHandler
  {
  public:
    /**
     * Constructor that stores a reference to the HeatData object.
     *
     * @param heat_data The HeatData object to be referenced.
     */
    explicit DoFHandler(const HeatData<dim, number> &heat_data);

    /**
     * Converts the given cartesian index to the corresponding array index for
     * the same DoF.
     *
     * @param cartesian_index The cartesian index to be converted.
     * @return The corresponding array index for the given cartesian index.
     */
    [[nodiscard]] std::size_t
    get_array_index(const std::array<std::size_t, dim> &cartesian_index) const;

    /**
     * Converts an index in array format to the corresponding cartesian
     * coordinates of the same DoF.
     *
     * @param index The index in array format to be converted.
     * @return The cartesian coordinates corresponding to the given index.
     */
    std::array<std::size_t, dim>
    get_cartesian_index(std::size_t index) const;

    /**
     * Checks whether the degree of freedom (DoF) identified by the given index
     * is at any of the domain's boundaries.
     *
     * @param index The index of the DoF to check.
     * @return True if the DoF is on a boundary, false otherwise.
     */
    bool
    at_boundary(std::size_t index) const;

    /**
     * Converts a cartesian index to the physical coordinates of the
     * corresponding DoF in the domain. The returned array contains the
     * coordinates in each direction.
     *
     * @param index The cartesian index of the DoF.
     * @return An array of physical coordinates corresponding to the given
     * index.
     */
    std::array<number, dim>
    get_physical_location(const std::array<std::size_t, dim> &index) const;


    /**
     * Converts an array index to the physical coordinates of the
     * corresponding DoF in the domain. The returned array contains the
     * coordinates in each direction.
     *
     * @param index The array index of the DoF.
     * @return An array of physical coordinates corresponding to the given
     * index.
     */
    std::array<number, dim>
    get_physical_location(std::size_t index) const;


    /**
     * @brief Initializes a parallel distributed vector for the heat equation
     * solver. This involves computing the locally owned and ghost index sets to
     * properly partition the degrees of freedom across MPI processes.
     *
     * If the number of MPI processes exceeds the number of nodes in the
     * splitting direction, the vector is initialized using a new MPI
     * communicator that matches the number of nodes along that direction.
     *
     * @param dof_vector   The distributed vector to initialize.
     * @param init_value   The initial value to assign to all vector entries.
     */
    void
    initialize_dof_vector(ParallelDistributedVector &dof_vector,
                          double                     init_value = 0.) const;

  private:
    const HeatData<dim, number> &heat_data;
  };
} // namespace i2phpp
