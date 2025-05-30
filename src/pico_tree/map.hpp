#pragma once

#include <type_traits>

#include "core.hpp"
#include "point_traits.hpp"

namespace pico_tree {

namespace internal {

template <typename Element_, size_t Extent_>
struct map_storage {
  constexpr map_storage(Element_* data, size_t) noexcept : data(data) {}

  Element_* data;
  static size_t constexpr size = Extent_;
};

template <typename Element_>
struct map_storage<Element_, dynamic_extent> {
  constexpr map_storage(Element_* data, size_t size) noexcept
      : data(data), size(size) {}

  Element_* data;
  size_t size;
};

template <typename Element_, size_t Extent_>
class map {
 public:
  static_assert(std::is_object_v<Element_>, "ELEMENT_NOT_AN_OBJECT_TYPE");
  static_assert(
      !std::is_abstract_v<Element_>, "ELEMENT_CANNOT_BE_AN_ABSTRACT_TYPE");
  static_assert(
      Extent_ == dynamic_extent || Extent_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using element_type = Element_;
  using value_type = std::remove_cv_t<Element_>;
  using size_type = size_t;
  static constexpr size_type extent = Extent_;

  explicit constexpr map(element_type* data) noexcept
      : storage_(data, extent) {}

  constexpr map(element_type* data, size_type size) noexcept
      : storage_(data, size) {}

  template <typename ContiguousAccessIterator_>
  constexpr map(
      ContiguousAccessIterator_ begin, ContiguousAccessIterator_ end) noexcept
      : storage_(&(*begin), static_cast<size_type>(end - begin)) {}

  constexpr element_type& operator[](size_type i) const {
    return storage_.data[i];
  }

  constexpr element_type* data() const noexcept { return storage_.data; }

  constexpr size_type size() const noexcept { return storage_.size; }

 protected:
  map_storage<Element_, Extent_> storage_;
};

template <typename Scalar_, size_t Dim_>
struct space_map_matrix_storage {
  constexpr space_map_matrix_storage(Scalar_* data, size_t size, size_t)
      : data(data), size(size) {}

  Scalar_* data;
  size_t size;
  static size_t constexpr sdim = Dim_;
};

template <typename Scalar_>
struct space_map_matrix_storage<Scalar_, dynamic_extent> {
  constexpr space_map_matrix_storage(Scalar_* data, size_t size, size_t sdim)
      : data(data), size(size), sdim(sdim) {}

  Scalar_* data;
  size_t size;
  size_t sdim;
};

}  // namespace internal

//! \brief The point_map class provides a point interface for an array of
//! scalars.
template <typename Scalar_, size_t Dim_>
class point_map : protected internal::map<Scalar_, Dim_> {
 private:
  using base = internal::map<Scalar_, Dim_>;

 public:
  static_assert(std::is_arithmetic_v<Scalar_>, "SCALAR_NOT_AN_ARITHMETIC_TYPE");

  using scalar_type = typename base::value_type;
  using typename internal::map<Scalar_, Dim_>::element_type;
  using typename internal::map<Scalar_, Dim_>::size_type;
  static size_type constexpr dim = base::extent;

  using internal::map<Scalar_, Dim_>::map;
  using internal::map<Scalar_, Dim_>::operator[];
  using internal::map<Scalar_, Dim_>::data;
  using internal::map<Scalar_, Dim_>::size;
};

//! \brief The space_map class provides a space interface for an array of
//! points.
template <typename Point_>
class space_map : protected internal::map<Point_, dynamic_extent> {
  using base = internal::map<Point_, dynamic_extent>;

 public:
  using point_type = typename base::value_type;
  using point_element_type = typename base::element_type;
  using scalar_type = typename point_traits<point_type>::scalar_type;
  using size_type = size_t;
  static size_type constexpr dim = point_traits<point_type>::dim;

  static_assert(
      dim != dynamic_extent, "SPACE_MAP_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");

  using internal::map<Point_, dynamic_extent>::operator[];
  using internal::map<Point_, dynamic_extent>::data;
  using internal::map<Point_, dynamic_extent>::size;

  constexpr space_map(point_element_type* data, size_type size) noexcept
      : internal::map<Point_, dynamic_extent>::map(data, size) {}

  constexpr size_type sdim() const { return dim; }
};

//! \brief The space_map class provides a space interface for an array of
//! scalars.
template <typename Scalar_, size_t Dim_>
class space_map<point_map<Scalar_, Dim_>> {
 public:
  using point_type = point_map<Scalar_, Dim_>;
  using scalar_type = typename point_type::scalar_type;
  using scalar_element_type = typename point_type::element_type;
  using size_type = typename point_type::size_type;
  static size_type constexpr dim = point_type::dim;

  constexpr space_map(scalar_element_type* data, size_type size) noexcept
      : storage_(data, size, dim) {}

  constexpr space_map(
      scalar_element_type* data, size_type size, size_type sdim) noexcept
      : storage_(data, size, sdim) {}

  constexpr point_type operator[](size_type i) const noexcept {
    return {data(i), storage_.sdim};
  }

  constexpr scalar_element_type* data() const noexcept { return storage_.data; }

  constexpr scalar_element_type* data(size_type i) const noexcept {
    return storage_.data + i * storage_.sdim;
  }

  constexpr size_type size() const noexcept { return storage_.size; }

  constexpr size_type sdim() const noexcept { return storage_.sdim; }

 protected:
  internal::space_map_matrix_storage<Scalar_, Dim_> storage_;
};

}  // namespace pico_tree
