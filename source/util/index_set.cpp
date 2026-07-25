#include <i2phpp/util/asserts.hpp>
#include <i2phpp/util/index_set.hpp>
#include <mpi.h>

#include <algorithm>
#include <string>

i2phpp::IndexSet::IndexSet(const unsigned n_indices)
{
  indices.resize(n_indices);
  for (unsigned i = 0; i < n_indices; ++i)
    indices[i] = i;
}

i2phpp::IndexSet::IndexSet(unsigned lower_bound, unsigned upper_bound)
{
  assert_release(
    upper_bound >= lower_bound,
    std::string(
      "An index set cannot have an upper bound that is smaller than the "
      "lower bound. Please check the input parameters! The provided lower "
      "bound is " +
      std::to_string(lower_bound) + " and the upper bound is " +
      std::to_string(upper_bound)),
    MPI_COMM_WORLD);

  indices.reserve(upper_bound - lower_bound);
  for (unsigned i = lower_bound; i < upper_bound; ++i)
    indices.emplace_back(i);
}

unsigned
i2phpp::IndexSet::size() const
{
  return indices.size();
}

unsigned
i2phpp::IndexSet::nth_index_in_set(const unsigned local_index) const
{
  return indices[local_index];
}

bool
i2phpp::IndexSet::is_element(const unsigned index) const
{
  if (is_contiguous())
    {
      return (index >= indices.front() and index <= indices.back());
    }
  return std::ranges::any_of(indices, [&](unsigned i) { return (i == index); });
}

void
i2phpp::IndexSet::add_index(const unsigned index)
{
  indices.push_back(index);
  sort();
}

const std::vector<unsigned> &
i2phpp::IndexSet::get_indices() const
{
  return indices;
}

bool
i2phpp::IndexSet::is_contiguous() const
{
  for (unsigned i = 1; i < indices.size(); ++i)
    {
      if (indices[i] != indices[i - 1] + 1)
        return false;
    }
  return true;
}

unsigned
i2phpp::IndexSet::index_within_set(const unsigned int global_index) const
{
  if (is_contiguous())
    {
      if (unsigned local_index = global_index - indices[0];
          global_index >= indices[0] && local_index < size())
        return global_index - indices[0];
    }
  else
    {
      for (unsigned i = 0; i < indices.size(); ++i)
        if (indices[i] == global_index)
          return i;
    }
  assert_release(false,
                 std::string("Index is not in index range!"),
                 MPI_COMM_WORLD);
  return -1;
}

typename std::vector<unsigned>::iterator
i2phpp::IndexSet::begin()
{
  return indices.begin();
}

typename std::vector<unsigned>::iterator
i2phpp::IndexSet::end()
{
  return indices.end();
}

void
i2phpp::IndexSet::sort()
{
  std::sort(indices.begin(), indices.end());
}
