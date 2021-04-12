//
// Created by Angel Dong on 4/11/21.
//

#ifndef NAIVE_BAYES_CLASSIFIER_H
#define NAIVE_BAYES_CLASSIFIER_H

#include "core/model.h"

namespace naivebayes {
  /**
   * Class that processes and classifies images.
   */
  class Classifier {

  public:
    std::vector<double> likelihood_;// vector of likelihood of class
    std::vector<size_t> predicted_class_; // vector of predicted class

    /**
     * Method that returns the predicted class of an image.
     * 
     * @param image to be predicted
     * @return predicted class of the image
     */
    size_t ReturnPredictedClass(Image &image, const Model &model);

    /**
     * Method that calculates the accuracy of the predictor.
     * 
     * @param model 
     * @return accuracy of predictor
     */
    double CalculateAccuracyPercentage(Model &model);
  };
}// namespace naivebayes
#endif//NAIVE_BAYES_CLASSIFIER_H
