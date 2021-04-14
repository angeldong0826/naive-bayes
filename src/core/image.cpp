#include "core/image.h"
#include <iostream>
#include <istream>
#include <string>

namespace naivebayes {

  Image::Image() {}

  Image::Image(size_t image_size) {
    image_size_ = image_size;
    grid_ = std::vector<std::vector<size_t>>(image_size_, std::vector<size_t>(image_size_));
  }

  std::istream &operator>>(std::istream &is, Image &image) {
    image.grid_ = std::vector<std::vector<size_t>>(image.image_size_, std::vector<size_t>(image.image_size_));

    size_t count = 0;
    for (std::string line; std::getline(is, line); count++) {
      for (int i = 0; i < line.length(); ++i) {

        if (line[i] == '+' || line[i] == '#') {
          image.grid_[count % image.image_size_][i] = 1;// shaded
        } else {
          image.grid_[count % image.image_size_][i] = 0;// unshaded
        }
      }
      if (count == image.image_size_ - 1) {// last line
        break;
      }
    }

    return is;
  }


  size_t Image::GetValue(size_t row, size_t col) const {
    return grid_[row][col];
  }

  void Image::SetValue(size_t row, size_t col, size_t value) {
    grid_[row][col] = value;
  }

  void Image::SetImageSize(size_t size) {
    image_size_ = size;
  }

  void Image::SetGridSize(size_t size) {
    grid_ = std::vector<std::vector<size_t>>(size, std::vector<size_t>(size));
  }
  
  void Image::SetGridVector(std::vector<std::vector<size_t>> grid) {
    for (size_t row = 0; row < image_size_; row++) {
      for (size_t col = 0; col < image_size_; col++) {
        grid_[row][col] = grid[row][col];
      }
    }
  }

}// namespace naivebayes