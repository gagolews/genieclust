#pragma once

namespace pico_tree {

//! \brief point_traits provides an interface for the different point types that
//! are supported by PicoTree.
//! \details Examples of how a point_traits can be created and used are linked
//! below.
//! \tparam Point_ Any of the point types supported by point_traits.
//! \see point_traits<Scalar_[Dim_]>
//! \see space_traits<std::vector<Point_, Allocator_>>
template <typename Point_>
struct point_traits;

}  // namespace pico_tree
