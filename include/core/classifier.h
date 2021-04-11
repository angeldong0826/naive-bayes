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

    /**
     * Method that returns the predicted class of an image.
     * 
     * @param image to be predicted
     * @return predicted class of the image
     */
    size_t ReturnPredictedClass(const Image &image, const Model &model);

    //    /**
    //     * Helper method that finds the class with the highest probability.
    //     *
    //     * @return class number that's with the highest probability
    //     */
    //    size_t FindHighestProbabilityClass();
  };
}// namespace naivebayes
#endif//NAIVE_BAYES_CLASSIFIER_H
