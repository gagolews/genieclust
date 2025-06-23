#pragma once

#include <algorithm>
#include <iterator>

#include "pico_tree/core.hpp"

namespace pico_tree::internal {

//! \brief Inserts \p item in O(n) time at the index for which \p comp first
//! holds true. The sequence must be sorted and remains sorted after insertion.
//! The last item in the sequence is overwritten / "pushed out".
//! \details The contents of the indices at which \p comp holds true are moved
//! to the next index. Thus, starting from the end of the sequence, each item[i]
//! gets replaced by item[i - 1] until \p comp results in false. The worst case
//! has n comparisons and n copies, traversing the entire sequence.
//! <p/>
//! This algorithm is used as the inner loop of insertion sort:
//! * https://en.wikipedia.org/wiki/Insertion_sort
template <
    typename RandomAccessIterator_,
    typename Compare_ = std::less<
        typename std::iterator_traits<RandomAccessIterator_>::value_type>>
inline void insert_sorted(
    RandomAccessIterator_ begin,
    RandomAccessIterator_ end,
    typename std::iterator_traits<RandomAccessIterator_>::value_type item,
    Compare_ comp = Compare_()) {
  std::advance(end, -1);
  for (; end > begin && comp(item, *std::prev(end)); --end) {
    *end = std::move(*std::prev(end));
  }
  // We update the inserted element outside of the loop. This is done for the
  // case where we didn't break, simply reaching the end of the loop. This
  // happens when we need to replace the first element in the sequence (the last
  // item encountered).
  *end = std::move(item);
}

//! \brief kd_tree search visitor for finding a single nearest neighbor.
template <typename Neighbor_>
class search_nn {
 public:
  using neighbor_type = Neighbor_;
  using index_type = typename Neighbor_::index_type;
  using scalar_type = typename Neighbor_::scalar_type;

  //! \private
  inline search_nn(neighbor_type& nn) : nn_{nn} {
    nn_.distance = std::numeric_limits<scalar_type>::max();
  }

  //! \brief Visit current point.
  inline void operator()(index_type const idx, scalar_type const dst) const {
    if (max() > dst) {
      nn_ = {idx, dst};
    }
  }

  //! \brief Maximum search distance with respect to the query point.
  inline scalar_type max() const { return nn_.distance; }

 private:
  neighbor_type& nn_;
};

//! \brief kd_tree search visitor for finding k nearest neighbors using an
//! insertion sort.
//! \details Even though insertion sort is a rather brute-force method for
//! maintaining a sorted sequence, the k nearest neighbors, it performs fast in
//! practice. This is likely due to points being reasonably ordered by the
//! kd_tree. The following strategies have been attempted:
//!  * std::vector::insert(std::lower_bound) - the predecessor of the current
//!  version.
//!  * std::push_heap(std::vector) and std::pop_heap(std::vector).
//!  * std::push_heap(std::vector) followed by a custom ReplaceFrontHeap once
//!  the heap reached size k. This is the fastest "priority queue" version so
//!  far. Even without sorting the heap it is still slower than maintaining a
//!  sorted sequence. Unsorted it does come close to the insertion sort.
//!  * Binary heap plus a heap sort seemed a lot faster than the Leonardo heap
//!  with smooth sort.
template <typename RandomAccessIterator_>
class search_knn {
 public:
  static_assert(
      std::is_base_of_v<
          std::random_access_iterator_tag,
          typename std::iterator_traits<
              RandomAccessIterator_>::iterator_category>,
      "EXPECTED_RANDOM_ACCESS_ITERATOR");

  using neighbor_type =
      typename std::iterator_traits<RandomAccessIterator_>::value_type;
  using index_type = typename neighbor_type::index_type;
  using scalar_type = typename neighbor_type::scalar_type;

  //! \private
  inline search_knn(RandomAccessIterator_ begin, RandomAccessIterator_ end)
      : begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->distance = std::numeric_limits<scalar_type>::max();
  }

  //! \brief Visit current point.
  inline void operator()(index_type const idx, scalar_type const dst) {
    if (max() > dst) {
      if (active_end_ < end_) {
        ++active_end_;
      }

      insert_sorted(begin_, active_end_, neighbor_type{idx, dst});
    }
  }

  //! \brief Maximum search distance with respect to the query point.
  inline scalar_type max() const { return std::prev(end_)->distance; }

 private:
  RandomAccessIterator_ begin_;
  RandomAccessIterator_ end_;
  RandomAccessIterator_ active_end_;
};

//! \brief kd_tree search visitor for finding all neighbors within a radius.
template <typename Neighbor_>
class search_radius {
 public:
  using neighbor_type = Neighbor_;
  using index_type = typename Neighbor_::index_type;
  using scalar_type = typename Neighbor_::scalar_type;

  //! \private
  inline search_radius(scalar_type const radius, std::vector<neighbor_type>& n)
      : radius_{radius}, n_{n} {
    n_.clear();
  }

  //! \brief Visit current point.
  inline void operator()(index_type const idx, scalar_type const dst) const {
    if (max() > dst) {
      n_.push_back({idx, dst});
    }
  }

  //! \brief Sort the neighbors by distance from the query point. Can be used
  //! after the search has ended.
  inline void sort() const { std::sort(n_.begin(), n_.end()); }

  //! \brief Maximum search distance with respect to the query point.
  inline scalar_type max() const { return radius_; }

 private:
  scalar_type radius_;
  std::vector<neighbor_type>& n_;
};

//! \brief Search visitor for finding an approximate nearest neighbor.
//! \details Tree nodes are skipped by scaling down the search distance,
//! possibly not visiting the true nearest neighbor. An approximate nearest
//! neighbor will at most be a factor of distance ratio e farther from the
//! query point than the true nearest neighbor: max_ann_distance =
//! true_nn_distance * e.
template <typename Neighbor_>
class search_approximate_nn {
 public:
  using neighbor_type = Neighbor_;
  using index_type = typename Neighbor_::index_type;
  using scalar_type = typename Neighbor_::scalar_type;

  //! \private
  inline search_approximate_nn(scalar_type const e, neighbor_type& nn)
      : e_inv_{scalar_type(1.0) / e}, nn_{nn} {
    nn_.distance = std::numeric_limits<scalar_type>::max();
  }

  //! \brief Visit current point.
  inline void operator()(index_type const idx, scalar_type const dst) const {
    scalar_type sdst = dst * e_inv_;
    if (max() > sdst) {
      nn_ = {idx, sdst};
    }
  }

  //! \brief Maximum search distance with respect to the query point.
  inline scalar_type max() const { return nn_.distance; }

 private:
  scalar_type e_inv_;
  neighbor_type& nn_;
};

//! \brief Search visitor for finding approximate nearest neighbors.
//! \see search_approximate_nn
//! \see search_knn
template <typename RandomAccessIterator_>
class search_approximate_knn {
 public:
  static_assert(
      std::is_base_of_v<
          std::random_access_iterator_tag,
          typename std::iterator_traits<
              RandomAccessIterator_>::iterator_category>,
      "EXPECTED_RANDOM_ACCESS_ITERATOR");

  using neighbor_type =
      typename std::iterator_traits<RandomAccessIterator_>::value_type;
  using index_type = typename neighbor_type::index_type;
  using scalar_type = typename neighbor_type::scalar_type;

  //! \private
  inline search_approximate_knn(
      scalar_type const e,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end)
      : e_inv_{scalar_type(1.0) / e},
        begin_{begin},
        end_{end},
        active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->distance = std::numeric_limits<scalar_type>::max();
  }

  //! \brief Visit current point.
  inline void operator()(index_type const idx, scalar_type const dst) {
    scalar_type sdst = dst * e_inv_;
    if (max() > sdst) {
      if (active_end_ < end_) {
        ++active_end_;
      }

      // Replace the current maximum for which the distance is scaled to be:
      // d = d / e.
      insert_sorted(begin_, active_end_, neighbor_type{idx, sdst});
    }
  }

  //! \brief Maximum search distance with respect to the query point.
  inline scalar_type max() const { return std::prev(end_)->distance; }

 private:
  scalar_type e_inv_;
  RandomAccessIterator_ begin_;
  RandomAccessIterator_ end_;
  RandomAccessIterator_ active_end_;
};

//! \brief kd_tree search visitor for finding the approximate neighbors within a
//! radius.
//! \see search_approximate_nn
//! \see search_radius
template <typename Neighbor_>
class search_approximate_radius {
 public:
  using neighbor_type = Neighbor_;
  using index_type = typename Neighbor_::index_type;
  using scalar_type = typename Neighbor_::scalar_type;

  //! \private
  inline search_approximate_radius(
      scalar_type const e,
      scalar_type const radius,
      std::vector<neighbor_type>& n)
      : e_inv_{scalar_type(1.0) / e}, radius_{radius * e_inv_}, n_{n} {
    n_.clear();
  }

  //! \brief Visit current point.
  inline void operator()(index_type const idx, scalar_type const dst) const {
    scalar_type sdst = dst * e_inv_;
    if (max() > sdst) {
      n_.push_back({idx, sdst});
    }
  }

  //! \brief Sort the neighbors by distance from the query point. Can be used
  //! after the search has ended.
  inline void sort() const { std::sort(n_.begin(), n_.end()); }

  //! \brief Maximum search distance with respect to the query point.
  inline scalar_type max() const { return radius_; }

 private:
  scalar_type e_inv_;
  scalar_type radius_;
  std::vector<neighbor_type>& n_;
};

}  // namespace pico_tree::internal
