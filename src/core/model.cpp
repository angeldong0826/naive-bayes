#include "core/model.h"
#include <fstream>
#include <string>

namespace naivebayes {

  std::vector<Image> Model::ParseImage(const std::string file) {
    std::ifstream my_file;
    my_file.open(file);


    std::vector<Image> images;
    int count = 0;
    Image current;

    for (std::string line; std::getline(my_file, line); count++) {

      //iterate through the string and put the chars in there
      for (int i = 0; i < line.length(); i++) {
        current.grid[count % kImageSize][i] = line[i];
      }

      if (count % kImageSize == kImageSize - 1) {
        current.NumberCodeImages(); //remove if not turning to black and white
        images.push_back(current);
      }
    }

    return images;
  }

  std::vector<size_t> Model::ParseLabel(const std::string file) {
    std::ifstream my_file;
    my_file.open(file);

    std::vector<size_t> labels;

    for (std::string line; std::getline(my_file, line);) {
      labels.push_back(std::stoi(line));
    }

    return labels;
  }
}

