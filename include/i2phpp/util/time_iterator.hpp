#pragma once

#include <i2phpp/util/json_utils.hpp>

namespace i2phpp
{
  template <typename number>
  struct TimeIteratorData
  {
    /**
     * Final simulation time.
     */
    number end_time;

    /**
     * Used time step size, whenever possible.
     */
    number default_time_increment;

    /**
     * Maximal number of time steps before the simulation is terminated even
     * though the final time has not been reached.
     */
    unsigned int n_maximal_time_steps = 100000;

    void
    parse_parameters(const nlohmann::json &json_file)
    {
      json_parse_field(json_file["time stepping"], "end time", end_time);
      json_parse_field(json_file["time stepping"],
                       "time increment",
                       default_time_increment);
    }
  };

  template <typename number>
  class TimeIterator
  {
  public:
    /**
     * Constructor. Initializes the iterator with the provided time iterator
     * data.
     *
     * @param data Reference to the TimeIteratorData containing necessary
     * simulation timing data.
     */
    explicit TimeIterator(const TimeIteratorData<number> &data);

    /**
     * Computes the time step size for the next time step. If the new time step
     * overshoots the simulation end time, it corrects the time step size.
     *
     * @return The time step size for the next time step.
     */
    inline number
    compute_next_time_increment();

    /**
     * Retrieves the current simulation time.
     *
     * @return The current simulation time.
     */
    [[nodiscard]] inline number
    get_current_time() const;

    /**
     * Retrieves the size of the current time step.
     *
     * @return The size of the current time step.
     */
    [[nodiscard]] inline number
    get_current_time_increment() const;

    /**
     * Retrieves the current time step number.
     *
     * @return The current time step number.
     */
    [[nodiscard]] inline unsigned int
    get_current_time_step_number() const;

  private:
    //! current simulation time
    number current_time;

    //! simulation time at the previous time step
    number old_time;

    //! current size of the time step
    number current_time_increment;

    //! size of the time step for the last computed time step
    number old_time_increment;

    //! number of time steps already computed
    unsigned int n_time_steps;

    const TimeIteratorData<number> &data;
  };
} // namespace i2phpp


/*****************************************************************************
 * Function definitions
 *****************************************************************************/

template <typename number>
i2phpp::TimeIterator<number>::TimeIterator(const TimeIteratorData<number> &data)
  : data(data)
{}

template <typename number>
inline number
i2phpp::TimeIterator<number>::compute_next_time_increment()
{
  n_time_steps += 1;

  // currently we directly use the increment passed by the user
  current_time_increment = data.default_time_increment;

  // correct time increment if it would overshoot the end time
  if (current_time + current_time_increment > data.end_time)
    current_time_increment = data.end_time - current_time;

  // don't set old values for first time step
  if (n_time_steps > 1)
    {
      old_time           = current_time;
      old_time_increment = current_time_increment;
    }

  current_time += current_time_increment;

  return current_time_increment;
}

template <typename number>
inline number
i2phpp::TimeIterator<number>::get_current_time() const
{
  return current_time;
}

template <typename number>
inline number
i2phpp::TimeIterator<number>::get_current_time_increment() const
{
  return current_time_increment;
}

template <typename number>
inline unsigned int
i2phpp::TimeIterator<number>::get_current_time_step_number() const
{
  return n_time_steps;
}
