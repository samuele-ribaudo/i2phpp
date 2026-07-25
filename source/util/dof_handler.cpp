#include <i2phpp/util/asserts.hpp>
#include <i2phpp/util/dof_handler.hpp>

#include <algorithm>
#include <ranges>
#include <string>

template <int dim, typename number>
i2phpp::DoFHandler<dim, number>::DoFHandler(
  const HeatData<dim, number> &heat_data)
  : heat_data(heat_data)
{}


template <int dim, typename number>
std::size_t
i2phpp::DoFHandler<dim, number>::get_array_index(
  const std::array<std::size_t, dim> &cartesian_index) const
{
  assert_debug(
    std::ranges::all_of(std::views::iota(0, dim),
                        [&](const unsigned i) {
                          return cartesian_index[i] < heat_data.grid.n_dofs[i];
                        }),
    "The index combination for computing the array index is not valid!");

  std::size_t array_index = 0;
  std::size_t stride      = 1;

  for (unsigned int i = 0; i < dim; ++i)
    {
      array_index += stride * cartesian_index[i];
      stride *= heat_data.grid.n_dofs[i];
    }

  return array_index;
}


template <int dim, typename number>
std::array<std::size_t, dim>
i2phpp::DoFHandler<dim, number>::get_cartesian_index(std::size_t index) const
{
  static_assert((dim > 0) && (dim < 4),
                "The implementation only supports 1 <= dimension <= 3!");

  if constexpr (dim == 1)
    {
      return {index};
    }
  else if constexpr (dim == 2)
    {
      return {index % heat_data.grid.n_dofs[0],
              index / heat_data.grid.n_dofs[0]};
    }
  else if constexpr (dim == 3)
    {
      // For 3D, first compute the z-coordinate (layer index) by dividing the
      // total index by the total number of DOFs in a two-dimensional layer.
      std::size_t z_index =
        index / (heat_data.grid.n_dofs[0] * heat_data.grid.n_dofs[1]);

      // Subtract the number of DOFs in the lower layers to get the
      // 2D-equivalent index.
      index -= z_index * (heat_data.grid.n_dofs[0] * heat_data.grid.n_dofs[1]);

      // For the remaining 2D index, calculate the x and y coordinates as
      // before.
      return {index % heat_data.grid.n_dofs[0],
              index / heat_data.grid.n_dofs[0],
              z_index};
    }
}

template <int dim, typename number>
bool
i2phpp::DoFHandler<dim, number>::at_boundary(std::size_t index) const
{
  std::array<std::size_t, dim> cartesian_indices = get_cartesian_index(index);
  for (unsigned int dimension = 0; dimension < dim; ++dimension)
    if (cartesian_indices[dimension] == 0 ||
        cartesian_indices[dimension] == heat_data.grid.n_dofs[dimension] - 1)
      return true;
  return false;
}


template <int dim, typename number>
std::array<number, dim>
i2phpp::DoFHandler<dim, number>::get_physical_location(
  const std::array<std::size_t, dim> &index) const
{
  assert_debug(
    std::ranges::all_of(std::views::iota(0, dim),
                        [&](const unsigned i) -> bool {
                          return index[i] < heat_data.grid.n_dofs[i];
                        }),
    "The index combination for computing the global index is not valid!");

  std::array<number, dim> coordinates{};
  for (unsigned int i = 0; i < dim; ++i)
    {
      coordinates[i] =
        static_cast<number>(index[i]) * heat_data.grid.delta_grid[i];
    }
  return coordinates;
}

template <int dim, typename number>
std::array<number, dim>
i2phpp::DoFHandler<dim, number>::get_physical_location(
  const std::size_t index) const
{
  return get_physical_location(get_cartesian_index(index));
}

template <int dim, typename number>
void
i2phpp::DoFHandler<dim, number>::initialize_dof_vector(
  i2phpp::ParallelDistributedVector &dof_vector,
  const double                       init_value) const
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int      rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // TODO

  unsigned int local_length = heat_data.grid.n_dofs[dim - 1] / size;

  unsigned end_offset = rank >= heat_data.grid.n_dofs[dim - 1] % size ?
                          heat_data.grid.n_dofs[dim - 1] % size :
                          rank + 1;

  unsigned begin_offset = rank >= heat_data.grid.n_dofs[dim - 1] % size ?
                            heat_data.grid.n_dofs[dim - 1] % size :
                            rank;

  // locally owned size
  unsigned local_size           = (local_length);
  unsigned elements_per_segment = 1;
  for (int dimension = dim - 2; dimension >= 0; --dimension)
    {
      elements_per_segment *= heat_data.grid.n_dofs[dimension];
      local_size *= heat_data.grid.n_dofs[dimension];
    }

  // local index set
  IndexSet locally_owned_indices(rank * local_size +
                                   begin_offset * elements_per_segment,
                                 (rank + 1) * local_size +
                                   end_offset * elements_per_segment);

  // ghost index set
  IndexSet ghost_indices;

  // Note in 1D n_y = n_z = 1, and in 2D n_z = 1
  if (rank == 0)
    for (int j = 0; j < heat_data.grid.n_dofs[1]; ++j)   // 3D
      for (int i = 0; i < heat_data.grid.n_dofs[0]; ++i) // 2D
        ghost_indices.add_index(local_size + end_offset * elements_per_segment +
                                j * heat_data.grid.n_dofs[1] + i);
  else if (rank == size - 1)
    for (int j = 0; j < heat_data.grid.n_dofs[1]; ++j)   // 3D
      for (int i = 0; i < heat_data.grid.n_dofs[0]; ++i) // 2D
        ghost_indices.add_index(rank * local_size +
                                begin_offset * elements_per_segment - 1 -
                                j * heat_data.grid.n_dofs[1] - i);
  else
    {
      for (int j = 0; j < heat_data.grid.n_dofs[1]; ++j)   // 3D
        for (int i = 0; i < heat_data.grid.n_dofs[0]; ++i) // 2D
          {
            ghost_indices.add_index(rank * local_size +
                                    begin_offset * elements_per_segment - 1 -
                                    j * heat_data.grid.n_dofs[1] - i);
            ghost_indices.add_index((rank + 1) * local_size +
                                    end_offset * elements_per_segment +
                                    j * heat_data.grid.n_dofs[1] + i);
          }
    }

  dof_vector.reinit(locally_owned_indices, ghost_indices, init_value, comm);
}


template class i2phpp::DoFHandler<1, double>;
template class i2phpp::DoFHandler<2, double>;
template class i2phpp::DoFHandler<3, double>;

template class i2phpp::DoFHandler<1, float>;
template class i2phpp::DoFHandler<2, float>;
template class i2phpp::DoFHandler<3, float>;