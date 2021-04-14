#pragma once

#include "core/model.h"

namespace naivebayes {

  /**
   * Class that processes and classifies images.
   */
  class Classifier {
  public:
    /**
     * Constructor of classifier that takes in a model.
     * 
     * @param model 
     */
    Classifier(Model &model);

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

    std::vector<double> likelihood_;

  private:
    Model &model_;                       // instance of model
    // vector of likelihood of class
    std::vector<size_t> predicted_class_;// vector of predicted class
  };

}// namespace naivebayes
