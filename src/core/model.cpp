#include <core/model.h>
#include <fstream>
#include <iostream>
#include <string>


namespace naivebayes {
  
  void Model::ParseImages(std::string file_path) {
    if (file_path.empty()) {
      throw std::invalid_argument("Empty filepath");
    }

    std::ifstream read;
    read.open(file_path);

    if (!read.is_open()) {
      throw std::invalid_argument("Unable to open training images");
    }

    images_.clear();
    
    while (!read.eof()) { // while not reached the end of file
      std::string line;
      std::getline(read, line);
      
      labels_.push_back(std::stoi(line));
      
      Image image{};
      read >> image;
      images_.push_back(image);
    }

    read.close();
  }
  
  void Model::CalculateFeatureProbabilities(size_t row, size_t col, size_t desired_class, size_t desired_shade) {
    size_t count = 0;
    for (size_t label = 0; label < labels_.size(); label++) {
      if (label == desired_class && images_[label].grid[row][col] == desired_shade) {
        count++;
      }
    }

    feature_prob_[row][col][desired_class][desired_shade] = log((kSmoothingFactor + static_cast<double>(count)) /
                                                                 (2 * kSmoothingFactor * static_cast<double>(class_prob_[desired_class])));
  }

  void Model::TrainModel() {
    
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < kNumClasses; num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {

            CalculateFeatureProbabilities(row, col, num, shade);
            CalculatePriorProbabilities();
          }
        }
      }
    }
  }

  void Model::CalculatePriorProbabilities() {
    for (size_t num = 0; num < kNumClasses; num++) {
      size_t count = 0;
      for (size_t label : labels_) {
        if (label == num) {
          count++;
        }
      }
      class_prob_.push_back(count);
    }
    
    for (size_t i = 0; i < kNumClasses; i++) {
      prior_prob_.at(i) = log((kSmoothingFactor + static_cast<double>(class_prob_.at(i))) / kNumClasses * kSmoothingFactor + static_cast<double>(labels_.size()));
    }
  }

  std::ostream &operator<<(std::ostream &os, Model &model) {
    for (size_t num = 0; num < Model::kNumClasses; num++) {
      for (size_t shade = 0; shade < Model::kShadeCount; shade++) {
        for (size_t row = 0; row < kImageSize; row++) {
          for (size_t col = 0; col < kImageSize; col++) {
            os << model.feature_prob_[row][col][num][shade] << " ";
          }
          
          os << std::endl;
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
            model.feature_prob_[row][col][num][shade] = probability;
          }
        }
      }
    }

    my_file.close();
  }

}// namespace naivebayes