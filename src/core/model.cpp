#include <core/model.h>
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
    
    read.close();
    return images_;
  }

  std::vector<size_t> Model::ParseLabel(std::string file_path) {
//    if (file_path.empty()) {
//      std::cout << "Empty filepath" << std::endl;
//      exit(1);
//    }

//    if (!read.is_open()) {
//      std::cout << "Unable to open training labels_" << std::endl;
//      exit(1);
//    }
    
//    std::vector<size_t> labels;
//
//    for(std::string current_string; std::getline(read, current_string);) {
//      labels.push_back(std::stoi(current_string));
//    }

    std::ifstream read;
    read.open(file_path);
    
    size_t count = 0;
    
    for (std::string line; std::getline(read, line); count++) {
      if (count % kImageSize == 0) {
        labels_.push_back(std::stoi(line));
      }
    }

    return labels_;
  }

  double Model::CalculatePixelProbabilityAtLocation(size_t row, size_t col, size_t desired_class, size_t desired_shade) {
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

  void Model::SetProbabilityArray() {
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < kNumClasses; num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {

            probability_array_[row][col][num][shade] = CalculatePixelProbabilityAtLocation(row, col, num, shade);
          }
        }
      }
    }
  }

  double Model::CalculatePrior(size_t input) {
    std::vector<size_t> vector = ParseLabel(file);
    std::cout << vector.size() << std::endl;
    
    for (size_t i = 0; i < vector.size(); i++) {
      if (vector[i] == input) {
        class_array_[i]++;
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
  }
  
  void Model::LoadData() {
    std::ifstream read;
    read.open(file);
    
    std::string line;
    while(!read.eof()) {
      
    }
  }

  //  std::string Model::ProbabilityArrayToString() {
//    std::string current;
//    for (size_t i = 0; i < kImageSize; i++) {
//      for (size_t j = 0; j < kImageSize; j++) {
//        for (size_t k = 0; k < kNumClasses; k++) {
//          for (size_t l = 0; l < kShadeCount; l++) {
//
//            current.push_back(probability_array_[i][j][k][l]);
//          }
//        }
//      }
//    }
//    
//    return current;
//  }
//
//  void Model::SaveData(std::string file_path) {
//    std::ofstream my_file;
//    my_file.open(file_path, std::ios::out);
//
//    if (!my_file.is_open()) {
//      std::cout << "Unable to open training images_" << std::endl;
//      exit(1);
//    }
//    
//    my_file << ProbabilityArrayToString();
//    my_file.close();
//  }
  
}// namespace naivebayes