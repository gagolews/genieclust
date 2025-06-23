#pragma once

#include <array>
#include <limits>
#include <vector>

#include "pico_tree/core.hpp"
#include "pico_tree/internal/segment.hpp"

namespace pico_tree::internal {

//! \brief box_traits exposes metadata for each of the different box types.
//! \see box
//! \see box_map
template <typename Box_>
struct box_traits;

//! \brief box_base exposes various box utilities.
//! \details CRTP based base class for any of the box child classes.
//! \tparam Derived_ Any of the box child classes.
template <typename Derived_>
class box_base {
 public:
  using scalar_type = typename box_traits<Derived_>::scalar_type;
  using size_type = size_t;
  static constexpr size_type dim = box_traits<Derived_>::dim;
  static_assert(dim == dynamic_extent || dim > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  //! \brief Returns true if \p x is contained. A point on the edge is
  //! considered inside the box.
  constexpr bool contains(scalar_type const* x) const {
    // We use derived().size() which includes the constexpr part. Otherwise a
    // small trait needs to be written.
    for (size_type i = 0; i < derived().size(); ++i) {
      if (min(i) > x[i] || max(i) < x[i]) {
        return false;
      }
    }
    return true;
  }

  //! \brief Returns true if \p x is contained. When the input box is identical,
  //! it is considered contained.
  template <typename OtherDerived_>
  constexpr bool contains(box_base<OtherDerived_> const& x) const {
    return contains(x.min()) && contains(x.max());
  }

  //! \brief Sets the values of min and max to be an inverted maximum bounding
  //! box.
  //! \details The values for min and max are set to respectively the maximum
  //! and minimum possible values for integers or floating points. This is
  //! useful for growing a bounding box in combination with the Update function.
  constexpr void fill_inverse_max() {
    for (size_type i = 0; i < derived().size(); ++i) {
      min(i) = std::numeric_limits<scalar_type>::max();
      max(i) = std::numeric_limits<scalar_type>::lowest();
    }
  }

  //! \copydoc fill_inverse_max
  static constexpr Derived_ make_inverse_max(size_type size = dim) {
    Derived_ box(size);
    box.fill_inverse_max();
    return box;
  }

  //! \brief See which axis of the box is the longest.
  //! \param p_max_index Output parameter for the index of the longest axis.
  //! \param p_max_value Output parameter for the range of the longest axis.
  constexpr void max_side(
      size_type& p_max_index, scalar_type& p_max_value) const {
    p_max_value = std::numeric_limits<scalar_type>::lowest();

    for (size_type i = 0; i < derived().size(); ++i) {
      scalar_type const delta = max(i) - min(i);
      if (delta > p_max_value) {
        p_max_index = i;
        p_max_value = delta;
      }
    }
  }

  //! \brief Updates the min and/or max vectors of this box so that it can fit
  //! input point \p x.
  constexpr void fit(scalar_type const* x) {
    for (size_type i = 0; i < derived().size(); ++i) {
      if (x[i] < min(i)) {
        min(i) = x[i];
      }
      if (x[i] > max(i)) {
        max(i) = x[i];
      }
    }
  }

  //! \brief Updates the min and/or max vectors of this box so that it can fit
  //! input box \p x.
  template <typename OtherDerived_>
  constexpr void fit(box_base<OtherDerived_> const& x) {
    for (size_type i = 0; i < derived().size(); ++i) {
      if (x.min(i) < min(i)) {
        min(i) = x.min(i);
      }

      if (x.max(i) > max(i)) {
        max(i) = x.max(i);
      }
    }
  }

  //! \brief Returns a const reference to the derived class.
  constexpr Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

  //! \brief Returns a reference to the derived class.
  constexpr Derived_& derived() { return *static_cast<Derived_*>(this); }

  constexpr scalar_type const* min() const noexcept { return derived().min(); }

  constexpr scalar_type* min() noexcept { return derived().min(); }

  constexpr scalar_type const* max() const noexcept { return derived().max(); }

  constexpr scalar_type* max() noexcept { return derived().max(); }

  constexpr scalar_type const& min(size_type i) const noexcept {
    return derived().min(i);
  }

  constexpr scalar_type& min(size_type i) noexcept { return derived().min(i); }

  constexpr scalar_type const& max(size_type i) const noexcept {
    return derived().max(i);
  }

  constexpr scalar_type& max(size_type i) noexcept { return derived().max(i); }

  constexpr size_type size() const noexcept { return derived().size(); }

 protected:
  //! \private
  constexpr box_base() = default;

  //! \private
  constexpr box_base(box_base const&) = default;

  //! \private
  constexpr box_base(box_base&&) = default;

  //! \private
  constexpr box_base& operator=(box_base const&) = default;

  //! \private
  constexpr box_base& operator=(box_base&&) = default;
};

//! \brief Storage container for the box class.
template <typename Scalar_, size_t Dim_>
struct box_storage {
  constexpr explicit box_storage(size_t) {}

  constexpr Scalar_ const* min() const noexcept { return coords_min.data(); }

  constexpr Scalar_* min() noexcept { return coords_min.data(); }

  constexpr Scalar_ const* max() const noexcept { return coords_max.data(); }

  constexpr Scalar_* max() noexcept { return coords_max.data(); }

  std::array<Scalar_, Dim_> coords_min;
  std::array<Scalar_, Dim_> coords_max;
  static size_t constexpr size = Dim_;
};

//! \brief Storage container for the box class.
//! \details A specialization with a run-time known spatial dimension.
template <typename Scalar_>
struct box_storage<Scalar_, dynamic_extent> {
  constexpr explicit box_storage(size_t size) : coords(size * 2), size(size) {}

  constexpr Scalar_ const* min() const noexcept { return coords.data(); }

  constexpr Scalar_* min() noexcept { return coords.data(); }

  constexpr Scalar_ const* max() const noexcept { return coords.data() + size; }

  constexpr Scalar_* max() noexcept { return coords.data() + size; }

  std::vector<Scalar_> coords;
  size_t size;
};

//! \brief An axis aligned box represented by a min and max coordinate.
template <typename Scalar_, size_t Dim_>
class box : public box_base<box<Scalar_, Dim_>> {
 public:
  using scalar_type = Scalar_;
  using typename box_base<box<Scalar_, Dim_>>::size_type;
  static size_type constexpr dim = Dim_;

  using box_base<box<Scalar_, Dim_>>::box_base;

  constexpr box() : storage_(dim) {}

  constexpr explicit box(size_type size) : storage_(size) {}

  constexpr scalar_type const* min() const noexcept { return storage_.min(); }

  constexpr scalar_type* min() noexcept { return storage_.min(); }

  constexpr scalar_type const* max() const noexcept { return storage_.max(); }

  constexpr scalar_type* max() noexcept { return storage_.max(); }

  constexpr scalar_type const& min(size_type i) const {
    return storage_.min()[i];
  }

  constexpr scalar_type& min(size_type i) { return storage_.min()[i]; }

  constexpr scalar_type const& max(size_type i) const {
    return storage_.max()[i];
  }

  constexpr scalar_type& max(size_type i) { return storage_.max()[i]; }

  constexpr size_type size() const noexcept { return storage_.size; }

 private:
  box_storage<Scalar_, Dim_> storage_;
};

//! \brief Storage container for the box_map class.
template <typename Scalar_, size_t Dim_>
struct box_map_storage {
  constexpr box_map_storage(Scalar_* min, Scalar_* max, size_t)
      : min(min), max(max) {}

  Scalar_* min;
  Scalar_* max;
  static size_t constexpr size = Dim_;
};

//! \brief Storage container for the box_map class.
//! \details A specialization with a run-time known spatial dimension.
template <typename Scalar_>
struct box_map_storage<Scalar_, dynamic_extent> {
  constexpr box_map_storage(Scalar_* min, Scalar_* max, size_t size)
      : min(min), max(max), size(size) {}

  Scalar_* min;
  Scalar_* max;
  size_t size;
};

//! \brief An axis aligned box represented by a min and max coordinate. It maps
//! raw pointers.
template <typename Scalar_, size_t Dim_>
class box_map : public box_base<box_map<Scalar_, Dim_>> {
 public:
  using scalar_type = std::remove_cv_t<Scalar_>;
  using element_type = Scalar_;
  using typename box_base<box_map<Scalar_, Dim_>>::size_type;
  static size_type constexpr dim = Dim_;

  constexpr box_map(element_type* min, element_type* max)
      : storage_(min, max, dim) {}

  constexpr box_map(element_type* min, element_type* max, size_type size)
      : storage_(min, max, size) {}

  constexpr box_map(box_map const&) = delete;

  constexpr box_map& operator=(box_map const&) = delete;

  constexpr element_type* min() const noexcept { return storage_.min; }

  constexpr element_type* max() const noexcept { return storage_.max; }

  constexpr element_type& min(size_type i) const { return storage_.min[i]; }

  constexpr element_type& max(size_type i) const { return storage_.max[i]; }

  constexpr size_type size() const noexcept { return storage_.size; }

 private:
  box_map_storage<Scalar_, Dim_> storage_;
};

// TODO Perhaps this could be part of te box class hierarchy.
//! \brief An axis aligned box represented by a min and max coordinate that
//! supports a topological space. It maps raw pointers.
//! \details This class is only used for storing the query box during a
//! search_box call.
//!
//! The Metric_ template parameter is used to determine the type of space.
template <typename Scalar_, size_t Dim_, typename Metric_>
class metric_box_map {
 public:
  using scalar_type = std::remove_cv_t<Scalar_>;
  using element_type = Scalar_;
  using size_type = size_t;
  static size_type constexpr dim = Dim_;

  constexpr metric_box_map(element_type* min, element_type* max)
      : storage_(min, max, dim) {}

  constexpr metric_box_map(element_type* min, element_type* max, size_type size)
      : storage_(min, max, size) {}

  constexpr metric_box_map(metric_box_map const&) = delete;

  constexpr metric_box_map& operator=(metric_box_map const&) = delete;

  bool contains(scalar_type const* const p) const {
    for (std::size_t i = 0; i < size(); ++i) {
      if (!contains(min(i), max(i), p[i], static_cast<int>(i))) {
        return false;
      }
    }
    return true;
  }

  // The input box is always one of the cells of the kd_tree. This means that
  // the min coordinate of the input box is always smaller than its max
  // coordinate.
  template <typename OtherDerived_>
  bool contains(box_base<OtherDerived_> const& x) const {
    for (std::size_t i = 0; i < size(); ++i) {
      if (!contains(
              min(i),
              max(i),
              segment_r1<scalar_type>(x.min(i), x.max(i)),
              static_cast<int>(i))) {
        return false;
      }
    }
    return true;
  }

  constexpr element_type* min() const noexcept { return storage_.min; }

  constexpr element_type* max() const noexcept { return storage_.max; }

  constexpr element_type& min(size_type i) const { return storage_.min[i]; }

  constexpr element_type& max(size_type i) const { return storage_.max[i]; }

  constexpr size_type size() const noexcept { return storage_.size; }

 private:
  template <typename T_>
  bool contains(scalar_type min, scalar_type max, T_ const& v, int dim) const {
    bool contains;
    auto bd = [&](auto one_space) {
      contains = make_segment(min, max, one_space).contains(v);
    };
    metric_.apply_dim_space(dim, bd);
    return contains;
  }

  segment_r1<scalar_type> make_segment(
      scalar_type min, scalar_type max, one_space_r1) const {
    return segment_r1<scalar_type>(min, max);
  }

  segment_s1<scalar_type> make_segment(
      scalar_type min, scalar_type max, one_space_s1) const {
    return segment_s1<scalar_type>(min, max);
  }

  box_map_storage<Scalar_, Dim_> storage_;
  Metric_ metric_;
};

template <typename Scalar_, size_t Dim_>
struct box_traits<box<Scalar_, Dim_>> {
  using scalar_type = std::remove_cv_t<Scalar_>;
  static size_t constexpr dim = Dim_;
};

template <typename Scalar_, size_t Dim_>
struct box_traits<box_map<Scalar_, Dim_>> {
  using scalar_type = std::remove_cv_t<Scalar_>;
  static size_t constexpr dim = Dim_;
};

}  // namespace pico_tree::internal
