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

    while (!read.eof()) {// while not reached the end of file
      std::string line;
      std::getline(read, line);

      labels_.push_back(std::stoi(line));

      Image image{};
      read >> image;
      images_.push_back(image);
    }

    read.close();
  }

  void Model::CalculateFeatureProbabilities() {
    size_t idx = 0;

    for (Image &image : images_) {
      size_t class_num = labels_[idx];

      for (size_t row = 0; row < kImageSize; row++) {
        for (size_t col = 0; col < kImageSize; col++) {
          size_t shade = image.GetValue(row, col) - '0';
          feature_count_[row][col][class_num][shade]++;
        }
      }
      idx++;
    }

    for (size_t row = 0; row < kImageSize; row++) {
      for (size_t col = 0; col < kImageSize; col++) {
        for (size_t class_num = 0; class_num < kNumClasses; class_num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {

            feature_prob_[row][col][class_num][shade] = log((kSmoothingFactor + static_cast<double>(feature_count_[row][col][class_num][shade])) /
                                                            (2 * kSmoothingFactor + static_cast<double>(class_[class_num])));
          }
        }
      }
    }
  }

  void Model::TrainModel() {
    CalculatePriorProbabilities();
    CalculateFeatureProbabilities();
  }

  void Model::CalculatePriorProbabilities() {
    class_ = std::vector<size_t>(kNumClasses);     // set size to vector
    prior_prob_ = std::vector<double>(kNumClasses);// set size to vector

    for (size_t label : labels_) {
      class_[label]++;
    }

    for (size_t i = 0; i < kNumClasses; i++) {
      prior_prob_[i] = log((kSmoothingFactor + static_cast<double>(class_[i])) /
                           (kNumClasses * kSmoothingFactor + static_cast<double>(labels_.size())));
    }
  }

  std::ostream &operator<<(std::ostream &os, Model &model) {
    for (size_t num = 0; num < Model::kNumClasses; num++) {
      os << model.prior_prob_[num] << std::endl;
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

  void Model::LoadData(std::string file) {
    std::ifstream my_file;
    my_file.open(file);

    for (size_t i = 0; i < kFeatureCount; i++) {
      std::string prior;
      getline(my_file, prior);
      prior_prob_.push_back(stod(prior));

      for (size_t row = 0; row < kImageSize; row++) {
        for (size_t col = 0; col < kImageSize; col++) {
          for (size_t num = 0; num < kNumClasses; num++) {
            for (size_t shade = 0; shade < kShadeCount; shade++) {

              std::string feature;
              getline(my_file, feature);
              feature_prob_[row][col][num][shade] = std::stod(feature);
            }
          }
        }
      }

      my_file.close();
    }
  }

}// namespace naivebayes