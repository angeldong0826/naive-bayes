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
    Model model; // instance of model object to be used in classification
    
    std::vector<double> likelihood_; // vector of likelihood of class
    
    /**
     * Method that calculates the predicted class of an image.
     * 
     * @param image to be predicted
     * @return prediced class of the image
     */
    void SetLikelihoodVector(Image& image);
  };
}
#endif//NAIVE_BAYES_CLASSIFIER_H
