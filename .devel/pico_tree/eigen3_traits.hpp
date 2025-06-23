#pragma once

#include <Eigen/Core>

#include "core.hpp"
#include "point_traits.hpp"
#include "space_traits.hpp"

//! \file eigen3_traits.hpp
//! \brief Provides an interface for spaces and points when working with types
//! from Eigen3.
//! \details It supports space_traits<> for dynamic matrices and maps of dynamic
//! matrices, but not for fixed size matrices or maps of those. Fixed size
//! matrices are mostly useful when they are small. See section "Fixed vs.
//! Dynamic size" of the following link:
//! * https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
//!
//! point_traits<> are supported for any type of matrix or matrix map.

namespace pico_tree {

namespace internal {

//! \brief A trait that determines if Derived_ inherits from
//! Eigen::MatrixBase<>.
template <typename Derived_>
struct is_matrix_base : public std::is_base_of<
                            Eigen::MatrixBase<std::remove_cv_t<Derived_>>,
                            std::remove_cv_t<Derived_>> {};

template <typename T>
inline constexpr bool is_matrix_base_v = is_matrix_base<T>::value;

template <typename Derived_>
constexpr int eigen_vector_dim() {
  static_assert(
      (!Derived_::IsRowMajor && Derived_::ColsAtCompileTime == 1) ||
          (Derived_::IsRowMajor && Derived_::RowsAtCompileTime == 1),
      "DERIVED_TYPE_IS_NOT_A_VECTOR");
  return Derived_::IsRowMajor ? Derived_::ColsAtCompileTime
                              : Derived_::RowsAtCompileTime;
}

constexpr size_t eigen_dim_to_pico_dim(int dim) {
  return dim == Eigen::Dynamic ? dynamic_extent : static_cast<size_t>(dim);
}

//! \brief eigen_point_traits provides an interface for the different point
//! types that can be used with EigenTraits.
//! \details Unlike the specialization of point_traits for Eigen types, the
//! internal implementation supports matrix expressions.
template <typename Derived_>
struct eigen_point_traits {
  static_assert(
      is_matrix_base_v<Derived_>, "DERIVED_TYPE_IS_NOT_AN_EIGEN_MATRIX");
  //! \brief Supported point type.
  using point_type = Derived_;
  //! \brief The scalar type of point coordinates.
  using scalar_type = std::remove_cv_t<typename Derived_::Scalar>;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static size_type constexpr dim =
      eigen_dim_to_pico_dim(eigen_vector_dim<Derived_>());

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static scalar_type const* data(Derived_ const& point) {
    return point.derived().data();
  }

  //! \brief Returns the spatial dimension of \p point.
  inline static size_type size(Derived_ const& point) {
    return static_cast<size_type>(point.size());
  }
};

template <typename Derived_>
struct eigen_traits_base {
  static_assert(
      Derived_::RowsAtCompileTime == Eigen::Dynamic ||
          Derived_::ColsAtCompileTime == Eigen::Dynamic,
      "FIXED_SIZE_MATRICES_ARE_NOT_SUPPORTED");

  //! \brief The space_type of these traits.
  using space_type = Derived_;
};

//! \brief Space and Point traits for Eigen types.
template <typename Derived_, bool RowMajor = Derived_::IsRowMajor>
struct eigen_traits_impl;

//! \brief Space and Point traits for ColMajor Eigen types.
template <typename Derived_>
struct eigen_traits_impl<Derived_, false> : public eigen_traits_base<Derived_> {
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Spatial dimension.
  static size_type constexpr dim =
      eigen_dim_to_pico_dim(Derived_::RowsAtCompileTime);
  //! \brief The point type used by Derived_.
  using point_type =
      Eigen::Block<Derived_ const, Derived_::RowsAtCompileTime, 1, true>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = std::remove_cv_t<typename Derived_::Scalar>;

  //! \brief Returns the point at index \p idx.
  template <typename Index_>
  inline static point_type point_at(Derived_ const& matrix, Index_ idx) {
    return matrix.col(static_cast<Eigen::Index>(idx));
  }

  //! \brief Returns the number of points.
  inline static size_type size(Derived_ const& matrix) {
    return static_cast<size_type>(matrix.cols());
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static size_type sdim(Eigen::MatrixBase<Derived_> const& matrix) {
    return static_cast<size_type>(matrix.rows());
  }
};

//! \brief Space and Point traits for RowMajor Eigen types.
template <typename Derived_>
struct eigen_traits_impl<Derived_, true> : public eigen_traits_base<Derived_> {
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Spatial dimension.
  static size_type constexpr dim =
      eigen_dim_to_pico_dim(Derived_::ColsAtCompileTime);
  //! \brief The point type used by Derived_.
  using point_type =
      Eigen::Block<Derived_ const, 1, Derived_::ColsAtCompileTime, true>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = std::remove_cv_t<typename Derived_::Scalar>;

  //! \brief Returns the point at index \p idx.
  template <typename Index_>
  inline static point_type point_at(Derived_ const& matrix, Index_ idx) {
    return matrix.row(static_cast<Eigen::Index>(idx));
  }

  //! \brief Returns the number of points.
  inline static size_type size(Derived_ const& matrix) {
    return static_cast<size_type>(matrix.rows());
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static size_type sdim(Derived_ const& matrix) {
    return static_cast<size_type>(matrix.cols());
  }
};

}  // namespace internal

//! \brief EigenTraits provides an interface for Eigen::Matrix<>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_>
struct space_traits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : public internal::eigen_traits_impl<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
};

//! \brief EigenTraits provides an interface for Eigen::Map<Eigen::Matrix<>>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct space_traits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    MapOptions_,
    StrideType_>>
    : public internal::eigen_traits_impl<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

//! \brief EigenTraits provides an interface for Eigen::Map<Eigen::Matrix<>
//! const>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct space_traits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> const,
    MapOptions_,
    StrideType_>>
    : public internal::eigen_traits_impl<Eigen::Map<
          Eigen::
              Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> const,
          MapOptions_,
          StrideType_>> {};

//! \brief point_traits provides an interface for Eigen::Matrix<>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_>
struct point_traits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : public internal::eigen_point_traits<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
};

//! \brief point_traits provides an interface for Eigen::Map<Eigen::Matrix<>>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct point_traits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    MapOptions_,
    StrideType_>>
    : public internal::eigen_point_traits<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

//! \brief point_traits provides an interface for Eigen::Map<Eigen::Matrix<>
//! const>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct point_traits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> const,
    MapOptions_,
    StrideType_>>
    : public internal::eigen_point_traits<Eigen::Map<
          Eigen::
              Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> const,
          MapOptions_,
          StrideType_>> {};

//! \brief point_traits provides an interface for Eigen::Block<>.
template <typename XprType_, int BlockRows_, int BlockCols_, bool InnerPanel_>
struct point_traits<Eigen::Block<XprType_, BlockRows_, BlockCols_, InnerPanel_>>
    : public internal::eigen_point_traits<
          Eigen::Block<XprType_, BlockRows_, BlockCols_, InnerPanel_>> {};

}  // namespace pico_tree
