#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/internal/box.hpp"
#include "pico_tree/space_traits.hpp"

namespace pico_tree::internal {

//! \brief The space_wrapper class wraps makes working with any space type
//! through its respective space_traits a bit easier and it allows for the
//! addition of extra convenience methods.
//! \details The internals of PicoTree never use the specializations of the
//! space_traits class directly, but interface with any space type through this
//! wrapper interface
template <typename Space_>
class space_wrapper {
  using space_traits_type = space_traits<Space_>;
  using space_type = Space_;
  using point_type = typename space_traits_type::point_type;
  using point_traits_type = point_traits<point_type>;

 public:
  using scalar_type = typename space_traits_type::scalar_type;
  using size_type = size_t;
  static size_type constexpr dim = space_traits_type::dim;

  inline explicit space_wrapper(space_type const& space) : space_(space) {}

  template <typename Index_>
  inline scalar_type const* operator[](Index_ const index) const {
    return point_traits_type::data(space_traits_type::point_at(space_, index));
  }

  inline box<scalar_type, dim> compute_bounding_box() const {
    auto bbox = box<scalar_type, dim>::make_inverse_max(sdim());
    for (size_type i = 0; i < size(); ++i) {
      bbox.fit(operator[](i));
    }
    return bbox;
  }

  inline size_type size() const { return space_traits_type::size(space_); }

  constexpr size_type sdim() const {
    if constexpr (dim != dynamic_extent) {
      return dim;
    } else {
      return space_traits_type::sdim(space_);
    }
  }

 private:
  space_type const& space_;
};

}  // namespace pico_tree::internal
