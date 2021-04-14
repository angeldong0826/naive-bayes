#include <core/model.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace naivebayes {

  Model::Model(size_t size) {
    image_size_ = size;

    feature_prob_ = std::vector<std::vector<std::vector<std::vector<double>>>>(image_size_,
                 std::vector<std::vector<std::vector<double>>>(image_size_,
                 std::vector<std::vector<double>>(kNumClasses,
                 std::vector<double>(kShadeCount))));

    feature_count_ = std::vector<std::vector<std::vector<std::vector<size_t>>>>(image_size_,
                  std::vector<std::vector<std::vector<size_t>>>(image_size_,
                  std::vector<std::vector<size_t>>(kNumClasses,
                  std::vector<size_t>(kShadeCount))));
  }

  void Model::ParseImages(std::string &file_path) {
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
      std::getline(read, line);// labels are in first line

      labels_.push_back(std::stoi(line));// push back parse label to labels vector to store as one unit

      Image image(image_size_);
      read >> image;           // rest of the lines are image- use image overloading operator to process
      images_.push_back(image);// push back parsed image to images vector to store as one unit
    }

    read.close();
  }

  void Model::CalculateFeatureProbabilities() {
    size_t idx = 0;

    for (Image &image : images_) {
      size_t class_num = labels_[idx];

      for (size_t row = 0; row < image_size_; row++) {
        for (size_t col = 0; col < image_size_; col++) {
          size_t shade = image.GetValue(row, col);
          feature_count_[row][col][class_num][shade]++;// counts number of feature at index
        }
      }
      idx++;
    }

    for (size_t row = 0; row < image_size_; row++) {
      for (size_t col = 0; col < image_size_; col++) {
        for (size_t class_num = 0; class_num < kNumClasses; class_num++) {
          for (size_t shade = 0; shade < kShadeCount; shade++) {

            feature_prob_[row][col][class_num][shade] = log((kSmoothingFactor + static_cast<double>(feature_count_[row][col][class_num][shade])) /
                                                            (2 * kSmoothingFactor + static_cast<double>(class_count_[class_num])));
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
    class_count_ = std::vector<size_t>(kNumClasses);// set size to vector
    prior_prob_ = std::vector<double>(kNumClasses); // set size to vector

    for (size_t label : labels_) {
      class_count_[label]++;// counts number of images in class
    }

    for (size_t i = 0; i < kNumClasses; i++) {
      prior_prob_[i] = log((kSmoothingFactor + static_cast<double>(class_count_[i])) /
                           (kNumClasses * kSmoothingFactor + static_cast<double>(labels_.size())));
    }
  }

  std::ostream &operator<<(std::ostream &os, Model &model) {

    for (size_t num = 0; num < kNumClasses; num++) {
      os << model.prior_prob_[num] << std::endl; // first line is always the prior probability

      for (size_t shade = 0; shade < kShadeCount; shade++) {
        for (size_t row = 0; row < model.image_size_; row++) {
          for (size_t col = 0; col < model.image_size_; col++) {
            os << model.feature_prob_[row][col][num][shade] << " "; // starting from second to image size store feature probability
          }

          os << std::endl;
        }
      }
    }

    return os;
  }

  std::istream &operator>>(std::istream &is, Model &model) {

    for (size_t num = 0; num < kNumClasses; num++) {
      std::string prior;
      getline(is, prior);
      model.prior_prob_.push_back(stod(prior)); // first line is prior probability

      for (size_t shade = 0; shade < kShadeCount; shade++) {

        for (size_t row = 0; row < model.image_size_; row++) {
          std::string feature;
          getline(is, feature);
          std::stringstream line_stream(feature);

          for (size_t col = 0; col < model.image_size_; col++) {
            line_stream >> feature;
            model.feature_prob_[row][col][num][shade] = std::stod(feature); // starting from second to image size store feature probability
          }
        }
      }
    }
    return is;
  }

  std::vector<Image> Model::GetImages() const {
    return images_;
  }

  std::vector<size_t> Model::GetLabels() const {
    return labels_;
  }

  size_t Model::GetImageSize() const {
    return image_size_;
  }

}// namespace naivebayes