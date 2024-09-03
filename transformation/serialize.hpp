/** Contains code to serialize the information collected on conditional branches
 *
 *  Copyright 2023, 2024 Philipp Andelfinger
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 *  and associated documentation files (the “Software”), to deal in the Software without
 *  restriction, including without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *   
 *    The above copyright notice and this permission notice shall be included in all copies or
 *    substantial portions of the Software.
 *    
 *    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>

void serialize(const std::unordered_map<std::string, std::vector<int>>& map, const std::string& filename) {
  std::ofstream ofs(filename, std::ios::binary);
  size_t map_size = map.size();
  ofs.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));

  for (const auto& pair : map) {
    size_t key_size = pair.first.size();
    ofs.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
    ofs.write(pair.first.data(), key_size);

    size_t vec_size = pair.second.size();
    ofs.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
    ofs.write(reinterpret_cast<const char*>(pair.second.data()), vec_size * sizeof(int));
  }
  ofs.close();
}

std::unordered_map<std::string, std::vector<int>> deserialize(const std::string& filename) {
  std::unordered_map<std::string, std::vector<int>> map;
  std::ifstream ifs(filename, std::ios::binary);

  size_t map_size;
  ifs.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));

  for (size_t i = 0; i < map_size; ++i) {
    size_t key_size;
    ifs.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
    
    std::string key(key_size, ' ');
    ifs.read(&key[0], key_size);

    size_t vec_size;
    ifs.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
    
    std::vector<int> vec(vec_size);
    ifs.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(int));

    map[key] = std::move(vec);
  }
  ifs.close();
  return map;
}
