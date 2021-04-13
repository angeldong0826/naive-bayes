#pragma once

#include "core/model.h"

namespace naivebayes {
  /**
   * Class that processes and classifies images.
   */
  class Classifier {

  public:
    explicit Classifier(Model &model);
    
    std::vector<double> likelihood_;// vector of likelihood of class
    std::vector<size_t> predicted_class_; // vector of predicted class

    /**
     * Method that returns the predicted class of an image_.
     * 
     * @param image to be predicted
     * @return predicted class of the image_
     */
    int ReturnPredictedClass(Image &image);

    /**
     * Method that calculates the accuracy of the predictor.
     * 
     * @param model 
     * @return accuracy of predictor
     */
    double CalculateAccuracyPercentage(Model &model);
    
  private:
    Model &model_;
  };
  
}// namespace naivebayes
