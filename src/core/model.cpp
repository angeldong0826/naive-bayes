#include "core/model.h"
#include <fstream>
#include <iostream>
#include <string>


namespace naivebayes {

  std::vector<Image> Model::ParseImage(const std::string file_path) {
    if (file_path.empty()) {
      std::cout << "Empty filepath" << std::endl;
      exit(1);
    }

    std::ifstream read;
    read.open(file_path);

    if (!read.is_open()) {
      std::cout << "Unable to open training images_" << std::endl;
      exit(1);
    }

    while (!read.eof()) { // while not reached the end of file
      Image image{};
      read >> image;
      images_.push_back(image);
    }

    return images_;
  }

  std::vector<size_t> Model::ParseLabel(std::string file_path) {
    if (file_path.empty()) {
      std::cout << "Empty filepath" << std::endl;
      exit(1);
    }

    std::ifstream read;
    read.open(file_path);

    if (!read.is_open()) {
      std::cout << "Unable to open training labels_" << std::endl;
      exit(1);
    }

    for (std::string line; std::getline(read, line);) {
      labels_.push_back(std::stoi(line));
    }

    return labels_;
  }

  //todo: DELETE
  //  std::string Model::GetBestClass() const {
  //    return "CS 126";
  //  }

  double Model::CalculateShadeProbability(size_t row, size_t col, size_t desired_class, size_t desired_shade) {
    size_t count = 0;
    size_t class_count = 0;

    for (size_t i = 0; i < images_.size(); i++) {
      if (labels_.at(i) == desired_class) {
        if (images_.at(i).grid[row][col] == desired_shade) {
          count++;
        }
      }
    }

    for (size_t label : labels_) {
      if (label == desired_class) {
        class_count++;
      }
    }

    return (kSmoothingFactor + count) / (2 * kSmoothingFactor * class_count);
  }

  void Model::CalculatePixelProbability() {
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < kNumClasses; num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {
            CalculateShadeProbability(row, col, num, shade);
          }
        }
      }
    }
  }

  double CalculatePrior(size_t input) {
    std::vector<size_t> vector = Model::ParseLabel(file);

    for (size_t i = 0; i < vector.size(); i++) {
      if (vector[i] == input) {
        Model::class_array[i] += 1;
      }
    }

    return (Model::kSmoothingFactor + Model::class_array[input]) / (Model::kNumClasses + Model::kSmoothingFactor * vector.size());
  }
}// namespace naivebayes