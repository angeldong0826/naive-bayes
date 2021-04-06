#include "core/model.h"
#include <fstream>
#include <string>

namespace naivebayes {

  void Model::ParseImage(const std::string file_path) {
    if (file_path.empty()) {
      throw std::invalid_argument("Empty file path");
    }

    std::ifstream read;
    read.open(file_path);

    while (!read.eof()) { // while not reached the end of file
      Image image;
      read >> image;
      images.push_back(image);
    }
  }

  void Model::ParseLabel(std::string file_path) {
    if (file_path.empty()) {
      throw std::invalid_argument("Empty filepath");
    }

    std::ifstream read;
    read.open(file_path);

    for (std::string line; std::getline(read, line);) {
      labels.push_back(std::stoi(line));
    }
  }

  std::string Model::GetBestClass() const {
    return "CS 126";
  }
}

