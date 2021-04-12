//
// Created by Angel Dong on 4/11/21.
//

#include "core/classifier.h"
#include <iostream>
namespace naivebayes {

  size_t Classifier::ReturnPredictedClass(Image &image, const Model &model) {
    likelihood_.resize(Model::kNumClasses, 0);

    for (size_t num = 0; num < likelihood_.size(); num++) {
      double likelihood_prob = 0;// variable that keeps track of the probability to be pushed to likelihood_

      likelihood_prob += model.prior_prob_.at(num);

      for (size_t row = 0; row < kImageSize; row++) {
        for (size_t col = 0; col < kImageSize; col++) {
          size_t shade = image.GetValue(row, col) - '0';
          likelihood_prob += model.feature_prob_[row][col][num][shade];
        }
      }
      likelihood_[num] = likelihood_prob;
    }

    double max = likelihood_[0];
    size_t idx = 0;

    for (size_t i = 1; i < likelihood_.size(); i++) {
      if (likelihood_[i] > max) {
        max = likelihood_.at(i);
        idx = i;
      }
    }

    return idx;
  }

  double Classifier::CalculateAccuracyPercentage(Model &model) {
    predicted_class_.resize(model.images_.size(), 0);

    for (size_t i = 0; i < model.images_.size(); i++) {
      predicted_class_[i] = ReturnPredictedClass(model.images_[i], model);
    }

    size_t count = 0;
    for (size_t i = 0; i < predicted_class_.size(); i++) {
//      std::cout << predicted_class_[i] << " " << model.labels_[i] << std::endl;
      if (predicted_class_[i] == model.labels_[i]) {
        count++;
      }
    }
    std::cout << "count: " << count << std::endl;
    return static_cast<double>(count) / static_cast<double>(model.images_.size()) * 100;
  }

}// namespace naivebayes
