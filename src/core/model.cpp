#include <core/model.h>
#include <fstream>
#include <iostream>
#include <string>


namespace naivebayes {

  std::vector<Image> Model::ParseImage(std::string file_path) {
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
    
    read.close();
    return images_;
  }

  std::vector<size_t> Model::ParseLabel(const std::string &file_path) {
    labels_.clear();
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
    
    size_t count = 0;
    
    for (std::string line; std::getline(read, line); count++) {
      if (count % (kImageSize + 1) == 0) {
        labels_.push_back(std::stoi(line));
      }
    }

    return labels_;
  }

  double Model::CalculatePixelProbability(size_t row, size_t col, size_t desired_class, size_t desired_shade) {
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
    
    return log((kSmoothingFactor + count) / (2 * kSmoothingFactor * class_count));
  }

  void Model::CalculateFeatureProbability() {
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < kNumClasses; num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {

            probability_array_[row][col][num][shade] = CalculatePixelProbability(row, col, num, shade);
          }
        }
      }
    }
  }

  double Model::CalculatePrior(size_t input, std::string &file) {
    for (size_t i = 0; i < kNumClasses; ++i) {
      class_array_[i] = 0;
    }
    std::vector<size_t> vector = ParseLabel(file);
    
    for (size_t i = 0; i < vector.size(); i++) {
      if (vector[i] == input) {
        class_array_[input] += 1;
      }
    }

    return log((kSmoothingFactor + class_array_[input]) / (kNumClasses + kSmoothingFactor * vector.size()));
  }

  std::ostream &operator<<(std::ostream &os, Model &model) {
    
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < Model::kNumClasses; num++) {
          for (size_t shade = 0; shade < Model::kShadeCount; shade++) {
            os << model.probability_array_[row][col][num][shade] << "\n";
          }
        }
      }
    }
    
    return os;
  }
  
  void Model::LoadData(std::string file, Model &model) {
    std::ifstream my_file;
    my_file.open(file);

    double probability;
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < kNumClasses; num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {
            my_file >> probability;
            model.probability_array_[row][col][num][shade] = probability;
          }
        }
      }
    }
    
    my_file.close();
  }
  
//  std::vector<Image> Model::GetImages() const {
//    return images_;
//  }
//  
//  std::vector<size_t> Model::GetLabels() const {
//    return labels_;
//  }

}// namespace naivebayes