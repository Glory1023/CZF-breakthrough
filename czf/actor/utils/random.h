#pragma once
#include <limits>
#include <random>

namespace czf::actor::utils::random {
class Splitmix {
 public:
  using result_type = uint64_t;
  using seed_type = uint64_t;
  static constexpr seed_type default_seed = 1u;

  explicit Splitmix(seed_type value = default_seed) { seed(value); }
  explicit Splitmix(std::random_device &rd) { seed(rd); }

  static constexpr result_type(min)() { return 0; }
  static constexpr result_type(max)() {
    return std::numeric_limits<result_type>::max();
  }

  void seed(seed_type value = default_seed) { m_seed = value; }
  void seed(std::random_device &rd) {
    m_seed = static_cast<uint64_t>(rd()) << 31 | static_cast<uint64_t>(rd());
  }

  result_type operator()() {
    seed_type z = (m_seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
  }

  void discard(unsigned long long n) {
    for (unsigned long long i = 0; i < n; ++i) {
      operator()();
    }
  }

  friend bool operator==(Splitmix const &lhs, Splitmix const &rhs) {
    return lhs.m_seed == rhs.m_seed;
  }
  friend bool operator!=(Splitmix const &lhs, Splitmix const &rhs) {
    return lhs.m_seed != rhs.m_seed;
  }

 private:
  seed_type m_seed = default_seed;
};

class Xorshift {
 public:
  using result_type = uint32_t;
  using seed_type = uint64_t;
  static constexpr seed_type default_seed = 0xc1f651c67c62c6e0ull;

  explicit Xorshift(seed_type value = default_seed) { seed(value); }
  explicit Xorshift(std::random_device &rd) { seed(rd); }

  void seed(seed_type value = default_seed) { m_seed = value; }
  void seed(std::random_device &rd) {
    m_seed = static_cast<uint64_t>(rd()) << 31 | static_cast<uint64_t>(rd());
  }

  static constexpr result_type(min)() { return 0; }
  static constexpr result_type(max)() {
    return std::numeric_limits<result_type>::max();
  }

  result_type operator()() {
    seed_type result = m_seed * 0xd989bcacc137dcd5ull;
    m_seed ^= m_seed >> 11;
    m_seed ^= m_seed << 31;
    m_seed ^= m_seed >> 18;
    return static_cast<result_type>(result >> 32ull);
  }

  void discard(unsigned long long n) {
    for (unsigned long long i = 0; i < n; ++i) {
      operator()();
    }
  }

  friend bool operator==(Xorshift const &lhs, Xorshift const &rhs) {
    return lhs.m_seed == rhs.m_seed;
  }
  friend bool operator!=(Xorshift const &lhs, Xorshift const &rhs) {
    return lhs.m_seed != rhs.m_seed;
  }

 private:
  seed_type m_seed = default_seed;
};
}  // namespace czf::actor::utils::random
