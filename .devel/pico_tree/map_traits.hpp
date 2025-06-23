#pragma once

//! \file map_traits.hpp
//! \brief Provides an interface for spaces and points when working with raw
//! pointers.

#include "map.hpp"
#include "space_traits.hpp"

namespace pico_tree {

template <typename Scalar_, size_t Dim_>
struct point_traits<point_map<Scalar_, Dim_>> {
  using point_type = point_map<Scalar_, Dim_>;
  using scalar_type = typename point_type::scalar_type;
  using size_type = typename point_type::size_type;
  static size_type constexpr dim = Dim_;

  inline static scalar_type const* data(point_type const& point) {
    return point.data();
  }

  inline static size_type size(point_type const& point) { return point.size(); }
};

//! \brief MapTraits provides an interface for spaces and points when working
//! with a space_map.
template <typename Point_>
struct space_traits<space_map<Point_>> {
  using space_type = space_map<Point_>;
  using point_type = typename space_type::point_type;
  using scalar_type = typename space_type::scalar_type;
  using size_type = typename space_type::size_type;
  static size_type constexpr dim = space_type::dim;

  template <typename Index_>
  inline static decltype(auto) point_at(space_type const& space, Index_ idx) {
    return space[static_cast<size_type>(idx)];
  }

  inline static size_type size(space_type const& space) { return space.size(); }

  inline static size_type sdim(space_type const& space) { return space.sdim(); }
};

}  // namespace pico_tree
