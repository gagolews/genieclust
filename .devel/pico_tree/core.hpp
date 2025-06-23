#pragma once

//! \mainpage PicoTree is a C++ header only library for nearest neighbor
//! searches and range searches using a kd_tree.
//! \file core.hpp
//! \brief Contains various common utilities.

#include <type_traits>

namespace pico_tree {

//! \brief Size type used by PicoTree.
using size_t = std::size_t;

//! \brief This value can be used in any template argument that wants to know
//! the spatial dimension of the search problem when it can only be known at
//! run-time. In this case the dimension of the problem is provided by the point
//! adaptor.
inline size_t constexpr dynamic_extent = static_cast<size_t>(-1);

//! \brief A Neighbor is a point reference with a corresponding distance to
//! another point.
template <typename Index_, typename Scalar_>
struct neighbor {
  static_assert(std::is_integral_v<Index_>, "INDEX_NOT_AN_INTEGRAL_TYPE");
  static_assert(std::is_arithmetic_v<Scalar_>, "SCALAR_NOT_AN_ARITHMETIC_TYPE");

  //! \brief Index type.
  using index_type = Index_;
  //! \brief Distance type.
  using scalar_type = Scalar_;

  //! \brief Default constructor.
  //! \details Declaring a custom constructor removes the default one. With
  //! C++11 we can bring back the default constructor and keep this struct a POD
  //! type.
  constexpr neighbor() = default;
  //! \brief Constructs a Neighbor given an index and distance.
  constexpr neighbor(index_type idx, scalar_type dst) noexcept
      : index(idx), distance(dst) {}

  //! \brief Point index of the Neighbor.
  index_type index;
  //! \brief Distance of the Neighbor with respect to another point.
  scalar_type distance;
};

//! \brief Compares neighbors by distance.
template <typename Index_, typename Scalar_>
constexpr bool operator<(
    neighbor<Index_, Scalar_> const& lhs,
    neighbor<Index_, Scalar_> const& rhs) noexcept {
  return lhs.distance < rhs.distance;
}

}  // namespace pico_tree
