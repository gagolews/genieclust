#pragma once

#include <cassert>
#include <opencv2/core.hpp>

#include "map_traits.hpp"

//! \file opencv_traits.hpp
//! \brief Contains traits that provide OpenCV support for PicoTree.
//! \details The following is supported:
//! * cv::Vec_<> as a point type.
//! * cv::Mat as a space type.

namespace pico_tree {

//! \brief point_traits provides an interface for cv::Point_<>.
//! \details point_traits<cv::Point_<Scalar_>> violates the strict aliasing rule
//! by interpreting a struct of scalars as an array of scalars and using this
//! specialization is therefore UB. Note that this specialization will work in
//! practice but you have been warned. Don't use it to avoid UB.
template <typename Scalar_>
struct point_traits<cv::Point_<Scalar_>> {
  static_assert(sizeof(cv::Point_<Scalar_>) == sizeof(Scalar_[2]), "");

  //! \brief Supported point type.
  using point_type = cv::Point_<Scalar_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = 2;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static scalar_type const* data(cv::Point_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point_.
  static constexpr size_type size(cv::Point_<Scalar_> const&) { return dim; }
};

//! \brief point_traits provides an interface for cv::Point3_<>.
//! \details point_traits<cv::Point3_<Scalar_>> violates the strict aliasing
//! rule by interpreting a struct of scalars as an array of scalars and using
//! this specialization is therefore UB. Note that this specialization will work
//! in practice but you have been warned. Don't use it to avoid UB.
template <typename Scalar_>
struct point_traits<cv::Point3_<Scalar_>> {
  static_assert(sizeof(cv::Point3_<Scalar_>) == sizeof(Scalar_[3]), "");

  //! \brief Supported point type.
  using point_type = cv::Point3_<Scalar_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = 3;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* data(cv::Point3_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point3_.
  static constexpr size_type size(cv::Point3_<Scalar_> const&) { return dim; }
};

//! \brief point_traits provides an interface for cv::Vec<>.
template <typename Scalar_, int Dim_>
struct point_traits<cv::Vec<Scalar_, Dim_>> {
  //! \brief Supported point type.
  using point_type = cv::Vec<Scalar_, Dim_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = static_cast<size_type>(Dim_);

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* data(cv::Vec<Scalar_, Dim_> const& point) {
    return point.val;
  }

  //! \brief Returns the spatial dimension of a cv::Vec.
  static constexpr size_type size(cv::Vec<Scalar_, Dim_> const&) { return dim; }
};

//! \brief The opencv_mat_map class provides compile time properties to a
//! cv::Mat.
//! \see space_map<Scalar_, Dim_>
template <typename Scalar_, size_t Dim_>
class opencv_mat_map : private space_map<point_map<Scalar_, Dim_>> {
  using space_map<point_map<Scalar_, Dim_>>::space_map;

 public:
  using opencv_mat_type =
      std::conditional_t<std::is_const_v<Scalar_>, cv::Mat const, cv::Mat>;
  using typename space_map<point_map<Scalar_, Dim_>>::point_type;
  using typename space_map<point_map<Scalar_, Dim_>>::scalar_type;
  using space_map<point_map<Scalar_, Dim_>>::dim;
  using typename space_map<point_map<Scalar_, Dim_>>::size_type;

  using space_map<point_map<Scalar_, Dim_>>::operator[];
  using space_map<point_map<Scalar_, Dim_>>::data;
  using space_map<point_map<Scalar_, Dim_>>::size;
  using space_map<point_map<Scalar_, Dim_>>::sdim;

  inline opencv_mat_map(opencv_mat_type mat)
      : space_map<point_map<Scalar_, Dim_>>::space_map(
            mat.template ptr<Scalar_>(),
            static_cast<size_type>(mat.rows),
            mat.step1()),
        mat_(mat) {}

  inline operator opencv_mat_type&() const { return mat_; }

  inline opencv_mat_type& mat() const { return mat_; }

 private:
  opencv_mat_type mat_;
};

//! \brief Provides an interface for cv::Mat. Each row is considered a point.
//! \tparam Scalar_ Point coordinate type.
//! \tparam Dim_ The spatial dimension of each point. Set to
//! pico_tree::dynamic_extent when the dimension is only known at run-time.
template <typename Scalar_, size_t Dim_>
struct space_traits<opencv_mat_map<Scalar_, Dim_>> {
  //! \brief The space_type of these traits.
  using space_type = opencv_mat_map<Scalar_, Dim_>;
  //! \brief The point type used by space_type.
  using point_type = typename space_type::point_type;
  //! \brief The scalar type of point coordinates.
  using scalar_type = typename space_type::scalar_type;
  //! \brief The size and index type of point coordinates.
  using size_type = typename space_type::size_type;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = space_type::dim;

  //! \brief Returns the point at \p idx from \p space.
  template <typename Index_>
  inline static point_type point_at(space_type const& space, Index_ idx) {
    return space[static_cast<size_type>(idx)];
  }

  //! \brief Returns number of points contained by \p space.
  inline static size_type size(space_type const& space) { return space.size(); }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  static constexpr size_type sdim(space_type const& space) {
    return space.sdim();
  }
};

}  // namespace pico_tree
