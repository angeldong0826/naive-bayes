#include "core/image.h"
#include <iostream>
#include <istream>
#include <string>

namespace naivebayes {

  std::istream &operator>>(std::istream &is, Image &image) {
    //    std::string first;
    //    getline(is, first);

    size_t count = 0;
    for (std::string line; std::getline(is, line); count++) {
      for (int i = 0; i < line.length(); ++i) {

        if (line[i] == '+' || line[i] == '#') {
          image.grid[count % kImageSize][i] = '1';
        } else {
          image.grid[count % kImageSize][i] = '0';
        }
      }
      if (count == kImageSize - 1) {
        break;
      }
    }

    return is;
  }
  
  char Image::GetValue(size_t row, size_t col) {
    return grid[row][col];
  }

}// namespace naivebayes