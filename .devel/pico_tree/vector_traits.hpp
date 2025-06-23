#pragma once

#include <vector>

#include "core.hpp"
#include "point_traits.hpp"
#include "space_traits.hpp"

namespace pico_tree {

//! \brief Provides an interface for std::vector<> and points supported by
//! PointTraits.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \tparam Allocator_ Allocator type used by the std::vector.
template <typename Point_, typename Allocator_>
struct space_traits<std::vector<Point_, Allocator_>> {
  //! \brief The space_type of these traits.
  using space_type = std::vector<Point_, Allocator_>;
  //! \brief The point type used by space_type.
  using point_type = Point_;
  //! \brief The scalar type of point coordinates.
  using scalar_type = typename point_traits<Point_>::scalar_type;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static size_type constexpr dim = point_traits<Point_>::dim;

  static_assert(
      dim != dynamic_extent, "VECTOR_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");

  //! \brief Returns the point at \p idx from \p space.
  template <typename Index_>
  inline static Point_ const& point_at(space_type const& space, Index_ idx) {
    return space[static_cast<size_type>(idx)];
  }

  //! \brief Returns number of points contained by \p space.
  inline static size_type size(space_type const& space) { return space.size(); }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static size_type constexpr sdim(space_type const&) { return dim; }
};

}  // namespace pico_tree
