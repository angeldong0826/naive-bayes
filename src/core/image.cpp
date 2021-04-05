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
  return is;
}

std::ostream& operator<<(std::ostream& os, Image &image) {
  return os;
}

} // namespace naivebayes