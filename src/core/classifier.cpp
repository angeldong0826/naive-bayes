#include "core/classifier.h"

#include <iostream>

namespace naivebayes {

  Classifier::Classifier(Model &model) : model_(model) {}

  int Classifier::ReturnPredictedClass(Image &image) {
    likelihood_.resize(kNumClasses, 0);

    for (size_t num = 0; num < likelihood_.size(); num++) {
      double likelihood_prob = 0;                // variable that keeps track of the probability to be pushed to likelihood_
      likelihood_prob += model_.prior_prob_[num];// first add prior probability

      for (size_t row = 0; row < model_.GetImageSize(); row++) {
        for (size_t col = 0; col < model_.GetImageSize(); col++) {
          size_t shade = image.GetValue(row, col);
          likelihood_prob += model_.feature_prob_[row][col][num][shade];// next add individual feature probability
        }
      }
      likelihood_[num] = likelihood_prob;// set total probability as an index in likelihood vector
    }

    double max = likelihood_[0];
    int idx = 0;

    for (int i = 1; i < likelihood_.size(); i++) {
      if (likelihood_[i] > max) {
        max = likelihood_[i];
        idx = i;
      }
    }

    return idx;// returns index of maximum probability
  }

  double Classifier::CalculateAccuracyPercentage(Model &model) {
    predicted_class_.resize(model.GetImages().size(), 0);

    for (size_t i = 0; i < model.GetImages().size(); i++) {
      predicted_class_[i] = ReturnPredictedClass(model.GetImages()[i]);// adds predicted class to a vector for comparison
    }

    size_t count = 0;
    for (size_t i = 0; i < predicted_class_.size(); i++) {
      if (predicted_class_[i] == model.GetLabels()[i]) {// if index at predicted equals the label (actual index)
        count++;
      }
    }

    return static_cast<double>(count) / static_cast<double>(model.GetImages().size()) * 100;
  }

}// namespace naivebayes
