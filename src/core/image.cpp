#include "core/image.h"
#include <string>
#include <istream>

namespace naivebayes {

void Image::NumberCodeImages() {
  for (auto &i : grid) {
    for (char j : i) {
      if (j == ' ') {
        j = 0;
      } else if (j == '+' || j == '#') {
        j = 1;
      }
    }
  }
}

std::istream& operator>>(std::istream& is, Image &image) {
  size_t count = 0;
  Image current{};

  for (std::string line; std::getline(is, line); count++) {

    //iterate through the string and fill vector with the chars
    for (int i = 0; i < line.length(); i++) {
      current.grid[count % kImageSize][i] = line[i];
    }

    if (count % kImageSize == kImageSize - 1) {
      current.NumberCodeImages();
      image.images.push_back(current);
    }
  }

  return is;
}

} // namespace naivebayes