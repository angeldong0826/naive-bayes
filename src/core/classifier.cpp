//
// Created by Angel Dong on 4/11/21.
//

#include "core/classifier.h"
namespace naivebayes {
  
  void Classifier::SetLikelihoodVector(naivebayes::Image &image) {
    likelihood_ = std::vector<double>(model.prior_prob_.size()); // set size to likelihood vector
    
    for (size_t num = 0; num < model.prior_prob_.size(); num++) {
      double likelihood_prob = 0; // variable that keeps track of the probability to be pushed to likelihood_
      
      likelihood_prob += model.prior_prob_.at(num);
      
      for (size_t row = 0; row < kImageSize; row++) {
        for (size_t col = 0; col < kImageSize; col++) {
          size_t shade = image.grid[row][col] - '0';
          likelihood_prob += model.feature_prob_[row][col][num][shade];
        }
      }
      likelihood_.push_back(likelihood_prob);
    }
  }
}

