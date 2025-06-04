/*
 * PicoTree: a C++ header only library for fast nearest neighbor
 * and range searches using a KdTree.
 *
 * <https://github.com/Jaybro/pico_tree>
 *
 * Version 1.0.0 (c5f719837df9707ee12d94cb0108aa0c34bfe96f)
 *
 * Copyright (c) 2025 Jonathan Broere
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


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
