#pragma once

#include <mpi.h>

#include <concepts>
#include <ranges>
#include <vector>

namespace i2phpp
{
  class IndexSet
  {
  public:
    /**
     * @brief Default constructor. Creates an empty IndexSet.
     */
    IndexSet() = default;

    /**
     * @brief Constructs an IndexSet with indices [0, n_indices).
     *
     * @param n_indices The number of indices to include.
     */
    explicit IndexSet(unsigned n_indices);

    /**
     * @brief Constructs an IndexSet with indices in the range  [lower_bound,
     *        upper_bound).
     *
     * @param lower_bound The inclusive lower bound.
     * @param upper_bound The exclusive upper bound.
     *
     * @throws MPI_Abort if upper_bound < lower_bound.
     */
    IndexSet(unsigned lower_bound, unsigned upper_bound);

    /**
     * @brief Constructs an IndexSet from a given range of unsigned integers.
     *
     * @tparam R A range type containing unsigned values.
     * @param in The range of indices.
     */
    template <std::ranges::input_range R>
      requires std::same_as<std::ranges::range_value_t<R>, unsigned> or
               std::same_as<std::ranges::range_value_t<R>, unsigned>
    explicit IndexSet(const R &in)
    {
      indices = std::vector<unsigned>(in.begin(), in.end());
      sort();
    }

    /**
     * @brief Returns the number of indices in the set.
     *
     * @return Number of elements in the index set.
     */
    [[nodiscard]] unsigned
    size() const;

    /**
     * @brief Checks if the index set contains a contiguous range.
     *
     * @return True if the indices form a continuous sequence.
     */
    [[nodiscard]] bool
    is_contiguous() const;

    /**
     * @brief Checks if a specific index is part of the set.
     *
     * @param index The global index to check.
     * @return True if the index is in the set.
     */
    [[nodiscard]] bool
    is_element(unsigned index) const;

    /**
     * @brief Retrieves the global index corresponding to a local position.
     *
     * @param local_index The position in the set.
     * @return The global index at that position.
     */
    [[nodiscard]] unsigned
    nth_index_in_set(unsigned local_index) const;

    /**
     * @brief Adds a new index to the set and subsequently resorts the index
     * set.
     *
     * @param index The index to add.
     */
    void
    add_index(unsigned index);

    /**
     * @brief Gets a const reference to the underlying vector of indices.
     *
     * @return The vector of indices.
     */
    [[nodiscard]] const std::vector<unsigned> &
    get_indices() const;

    /**
     * @brief Returns the position of a global index within the set.
     *
     * @param global_index The global index to look up.
     * @return The local index in the set.
     *
     * @throws MPI_Abort if the index is not found.
     */
    [[nodiscard]] unsigned
    index_within_set(const unsigned global_index) const;


    /**
     * @brief Returns an iterator to the beginning of the index set.
     */
    typename std::vector<unsigned>::iterator
    begin();

    /**
     * @brief Returns an iterator to the end of the index set.
     */
    typename std::vector<unsigned>::iterator
    end();

  private:
    /**
     * @brief Sorts the indices in ascending order.
     */
    void
    sort();

    /// Stores the actual indices.
    mutable std::vector<unsigned> indices;
  };
} // namespace i2phpp
