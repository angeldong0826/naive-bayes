#include "core/model.h"
#include <fstream>
#include <iostream>
#include <string>


namespace naivebayes {

  std::vector<Image> Model::ParseImage(const std::string file_path) {
    if (file_path.empty()) {
      throw std::invalid_argument("Empty filepath");
    }

    std::vector<Image> images;

    std::ifstream read;
    read.open(file_path);

    if (!read.is_open()) {
      std::cout << "Unable to open training images" << std::endl;
      exit(1);
    }

    while (!read.eof()) { // while not reached the end of file
      Image image;
      read >> image;
      images.push_back(image);
    }

    return images;
  }

  std::vector<size_t> Model::ParseLabel(std::string file_path) {
    if (file_path.empty()) {
      throw std::invalid_argument("Empty filepath");
    }

    std::vector<size_t> labels;

    std::ifstream read;
    read.open(file_path);

    if (!read.is_open()) {
      std::cout << "Unable to open training labels" << std::endl;
      exit(1);
    }

    for (std::string line; std::getline(read, line);) {
      labels.push_back(std::stoi(line));
    }

    return labels;
  }

  std::string Model::GetBestClass() const {
    return "CS 126";
  }
}

