/**
 * @brief Helper functions for convenient json usage.
 */

#pragma once

#include <string>

#include <nlohmann/json.hpp>


namespace i2phpp
{
  template <typename T>
  void
  json_parse_field(const nlohmann::json &j, const std::string &key, T &target)
  {
    if (j.contains(key) && !(j[key].is_null()))
      {
        try
          {
            target = j.at(key).get<T>();
          }
        catch (const nlohmann::json::type_error &e)
          {
            // do nothing
          }
      }
  }
} // namespace i2phpp
