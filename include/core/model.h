#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include "core/image.h"
#include "ostream"

namespace naivebayes {

  /**
   * Class that models image training.
   */
  class Model {
  public:
    static const constexpr double kSmoothingFactor = 1.0; // la place smoothing factor
    static const size_t kNumClasses = 10; // number of classes that images_ can belong to
    static const size_t kShadeCount = 2; // number-coded shade. 0 being unshaded and 1 being shaded

    std::vector<Image> images_; // vector of individual training images
    std::vector<size_t> labels_; // vector of labels of training images

    std::vector<size_t> class_prob_; // vector of number of images that belong to a class
    std::vector<double> prior_prob_; // vector of priors at a class
    
    double feature_prob_[kImageSize][kImageSize][kNumClasses][kShadeCount];
    
    size_t class_count_ = 0; // number of images in a class
    
    // todo: make methods private
    // todo: redo javadocs
    /**
    * Method that parses images_ from data file_path.
    *
    * @param file_path of file to be parsed
    */
    void ParseImages(std::string file_path);
    
    /**
     * Method that trains the model / calculates the pixel probability for every pixel in an image.
     */
    void TrainModel();

    /**
    * Method that calculates the prior of desired_class class.
    */
    void CalculatePriorProbabilities();
    
    /**
    * Operator << overload that saves the data needed to classify images to a file.
    */
    friend std::ostream& operator<<(std::ostream& os, Model &model);    
    
    /**
     * Method that loads the data back into a file.
     */
    void LoadData(std::string file, Model &model);

    void CalculateFeatureProbabilities(size_t row, size_t col, size_t desired_class, size_t desired_shade);
    };
}

#endif//NAIVE_BAYES_MODEL_H