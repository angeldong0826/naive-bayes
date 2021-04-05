#include "core/model.h"
#include <fstream>
#include <string>

namespace naivebayes {

  std::vector<Image> Model::ParseImage(const std::string file_to_parse) {
    if (file_to_parse.size() == 0) {
      throw std::invalid_argument("Empty filepath");
    }

    std::ifstream my_file;
    my_file.open(file_to_parse);


    std::vector<Image> images;
    int count = 0;
    Image current{};

    for (std::string line; std::getline(my_file, line); count++) {

      //iterate through the string and fill vector with the chars
      for (int i = 0; i < line.length(); i++) {
        current.grid[count % kImageSize][i] = line[i];
      }

      if (count % kImageSize == kImageSize - 1) {
        current.NumberCodeImages();
        images.push_back(current);
      }
    }

    return images;
  }

  std::vector<size_t> Model::ParseLabel(std::string file_to_parse) {
    if (file_to_parse.size() == 0) {
      throw std::invalid_argument("Empty filepath");
    }

    std::ifstream my_file;
    my_file.open(file_to_parse);

    std::vector<size_t> labels;

    for (std::string line; std::getline(my_file, line);) {
      labels.push_back(std::stoi(line));
    }

    return labels;
  }
}

