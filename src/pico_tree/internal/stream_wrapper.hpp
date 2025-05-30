#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace pico_tree::internal {

//! \brief Returns an std::fstream given a filename.
//! \details Convenience function that throws an std::runtime_error in case it
//! is unable to open the stream.
inline std::fstream open_stream(
    std::string const& filename, std::ios_base::openmode mode) {
  std::fstream stream(filename, mode);

  if (!stream.is_open()) {
    throw std::runtime_error("unable to open file: " + filename);
  }

  return stream;
}

//! \brief The stream_wrapper class is an std::iostream wrapper that helps read
//! and write various simple data types.
class stream_wrapper {
 public:
  //! \brief Constructs a stream_wrapper using an input std::iostream.
  stream_wrapper(std::iostream& stream) : stream_(stream) {}

  //! \brief Reads a single value from the stream_wrapper.
  //! \tparam T_ Type of the value.
  template <typename T_>
  inline void read(T_& value) {
    stream_.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(T_)));
  }

  //! \brief Reads a vector of values from the stream_wrapper.
  //! \details Reads the size of the vector followed by all its elements.
  //! \tparam T_ Type of a value.
  template <typename T_>
  inline void read(std::vector<T_>& values) {
    typename std::vector<T_>::size_type size;
    read(size);
    values.resize(size);
    read(size, values.data());
  }

  //! \brief Reads a string from the stream_wrapper.
  //! \details Reads the size of the string followed by all its elements.
  inline void read(std::string& values) {
    typename std::string::size_type size;
    read(size);
    values.resize(size);
    read(size, values.data());
  }

  //! \brief Reads an array of values from the stream_wrapper.
  //! \tparam T_ Type of a value.
  template <typename T_>
  inline void read(std::size_t size, T_* values) {
    stream_.read(
        reinterpret_cast<char*>(values),
        static_cast<std::streamsize>(sizeof(T_) * size));
  }

  //! \brief Writes a single value to the stream_wrapper.
  //! \tparam T_ Type of the value.
  template <typename T_>
  inline void write(T_ const& value) {
    stream_.write(
        reinterpret_cast<char const*>(&value),
        static_cast<std::streamsize>(sizeof(T_)));
  }

  //! \brief Writes a vector of values to the stream_wrapper.
  //! \details Writes the size of the vector followed by all its elements.
  //! \tparam T_ Type of a value.
  template <typename T_>
  inline void write(std::vector<T_> const& values) {
    write(values.size());
    write(values.data(), values.size());
  }

  //! \brief Writes a string to the stream_wrapper.
  //! \details Writes the size of the string followed by all its elements.
  inline void write(std::string const& values) {
    write(values.size());
    write(values.data(), values.size());
  }

  //! \brief Writes an array of values to the stream_wrapper.
  //! \tparam T_ Type of a value.
  template <typename T_>
  inline void write(T_ const* values, std::size_t size) {
    stream_.write(
        reinterpret_cast<char const*>(values),
        static_cast<std::streamsize>(sizeof(T_) * size));
  }

 private:
  //! \brief Wrapped stream.
  std::iostream& stream_;
};

}  // namespace pico_tree::internal
