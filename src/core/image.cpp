#include "core/image.h"
#include <iostream>
#include <istream>
#include <string>

namespace naivebayes {

std::istream& operator>>(std::istream& is, Image &image) {
  std::string first;
  getline(is, first);

  size_t count = 0;
  for (std::string line; std::getline(is, line); count++) {
    
    //iterate through the string and fill vector with the chars
    for (int i = 0; i < line.length(); ++i) {
      
      if (line[i] == '+' || line[i] == '#') {
        image.grid[count % kImageSize][i] = '1';
      } else {
        image.grid[count % kImageSize][i] = '0';
      }
    }
  }

  return is;
}

} // namespace naivebayes