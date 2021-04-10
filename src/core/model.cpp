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

  double Model::CalculateFeatureProbability(size_t row, size_t col, size_t desired_class, size_t desired_shade) {
    size_t count = 0;

    for (size_t i = 0; i < labels_.size(); i++) {
      if (labels_[i] == desired_class) {
        if (images_[i].grid[row][col] == desired_shade) {
          count++;
        }
      }
    }

    CalculateClassFrequency(desired_class);

//    for (size_t label : labels_) {
//      if (label == desired_class) {
//        class_count++;
//      }
//    }

    return log((kSmoothingFactor + static_cast<double>(count)) / (2 * kSmoothingFactor * static_cast<double>(class_count_)));
  }

  void Model::Train() {
    
    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t num = 0; num < kNumClasses; num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {

            probability_array_[row][col][num][shade] = CalculateFeatureProbability(row, col, num, shade);
          }
        }
      }
    }
  }

  double Model::CalculatePrior(size_t input, std::string &file) {
//    for (size_t i = 0; i < kNumClasses; i++) {
//      class_[i] = 0;
//    }
//    std::vector<size_t> vector = ParseLabel(file);
//    ParseLabel(file);

//    for (size_t i = 0; i < labels_.size(); i++) {
//      if (labels_[i] == input) {
//        class_[input] += 1;
//      }
//    }
    CalculateClassFrequency(input);

    return log((kSmoothingFactor + static_cast<double>(class_count_) / kNumClasses * kSmoothingFactor + static_cast<double>(labels_.size())));
  }

  std::ostream &operator<<(std::ostream &os, Model &model) {
    for (size_t num = 0; num < Model::kNumClasses; num++) {
      for (size_t shade = 0; shade < Model::kShadeCount; shade++) {
        for (size_t row = 0; row < kImageSize; row++) {
          for (size_t col = 0; col < kImageSize; col++) {
            os << model.probability_array_[row][col][num][shade] << " ";
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
            model.probability_array_[row][col][num][shade] = probability;
          }
        }
      }
    }

    my_file.close();
  }
  
  void Model::CalculateClassFrequency(size_t input) {
    size_t count = 0;

    for (size_t label : labels_) {
      if (label == input) {
        count++;
      }
    }
    
    class_count_ = count;
  }

  //  std::vector<Image> Model::GetImages() const {
  //    return images_;
  //  }
  //
  //  std::vector<size_t> Model::GetLabels() const {
  //    return labels_;
  //  }

}// namespace naivebayes