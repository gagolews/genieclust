#pragma once

#include <functional>
#include <type_traits>

namespace pico_tree::internal {

template <typename T_>
struct remove_reference_wrapper {
  using type = T_;
};

template <typename T_>
struct remove_reference_wrapper<std::reference_wrapper<T_>> {
  using type = std::remove_cv_t<T_>;
};

template <typename T_>
using remove_reference_wrapper_t = typename remove_reference_wrapper<T_>::type;

}  // namespace pico_tree::internal
