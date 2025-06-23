#pragma once

#include <functional>
#include <type_traits>

namespace pico_tree {

//! \brief space_traits provides an interface for the different space types that
//! are supported by PicoTree.
//! \tparam Space_ Any of the space types supported by space_traits.
template <typename Space_>
struct space_traits;

//! \brief Provides an interface for std::reference_wrapper<Space_>.
//! \details If Space_ is already a reference type, such as with an Eigen::Map<>
//! or cv::Mat, then using this specialization won't make much sense.
//! \tparam Space_ Any of the space types supported by space_traits.
template <typename Space_>
struct space_traits<std::reference_wrapper<Space_>>
    : public space_traits<std::remove_const_t<Space_>> {
  //! \brief The space_type of these traits.
  //! \details This overrides the space_type of the base class. No other
  //! interface changes are required as an std::reference_wrapper can implicitly
  //! be converted to its contained reference, which is a reference to an object
  //! of the exact same type as that of the space_type of the base class.
  using space_type = std::reference_wrapper<Space_>;
};

}  // namespace pico_tree
