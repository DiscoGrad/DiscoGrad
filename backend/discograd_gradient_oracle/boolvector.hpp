#pragma once

#include <vector>
#include <cstdint>

using namespace std;

class BoolVector {
public:
  void append(bool b) {
    int b_offset = b_size % 64;
    if (b_offset == 0)
      vec.resize(vec.size() + 1, 0);
    vec.back() |= (uint64_t)b << (63 - b_offset);
    b_size++;
  }
  
  uint64_t abs_dist(BoolVector& other) {
    size_t max_size = max(vec.size(), other.vec.size());
    uint64_t r = 0;
    for (size_t i = 0; i < max_size; i++) {
      uint64_t own_v = i < vec.size() ? vec[i] : 0;
      uint64_t other_v = i < other.vec.size() ? other.vec[i] : 0;
  
  
      uint64_t x = own_v ^ other_v;
      //printf("xor at %lu: %lu, popcount: %d\n", i, x, __builtin_popcountll(x));
  
      r += __builtin_popcountll(x);
    }
    return r;
  }
  
  void print() {
    size_t i = 0;
    for (auto &u : vec) {
      printf("%lu\n", u);
      for (int b_offset = 0; b_offset < 64; b_offset++) {
        printf("%lu: %lu\n", i, (u >> (63 - b_offset)) & 1);
        i++;
      }
    }
  }
  
  size_t bool_size() { return b_size; }

private:
  vector<uint64_t> vec;
  size_t b_size = 0;
};
