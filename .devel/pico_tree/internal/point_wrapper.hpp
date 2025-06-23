#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/point_traits.hpp"

namespace pico_tree::internal {

//! \brief The point_wrapper class wraps makes working with any point type
//! through its respective point_traits a bit easier and it allows for the
//! addition of extra convenience methods.
//! \details The internals of PicoTree never use the specializations of the
//! point_traits class directly, but interface with any point type through this
//! wrapper interface.
template <typename Point_>
class point_wrapper {
  using point_traits_type = point_traits<Point_>;
  using point_type = Point_;

 public:
  using scalar_type = typename point_traits_type::scalar_type;
  using size_type = size_t;
  static constexpr size_type dim = point_traits_type::dim;

  inline explicit point_wrapper(point_type const& point) : point_(point) {}

  inline scalar_type const& operator[](std::size_t index) const {
    return data()[index];
  }

  inline auto begin() const { return data(); }

  inline auto end() const { return data() + size(); }

 private:
  inline scalar_type const* data() const {
    return point_traits_type::data(point_);
  }

  constexpr size_type size() const {
    if constexpr (dim != dynamic_extent) {
      return dim;
    } else {
      return point_traits_type::size(point_);
    }
  }

  point_type const& point_;
};

}  // namespace pico_tree::internal
