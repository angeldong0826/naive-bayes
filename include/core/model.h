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
    static const constexpr double kSmoothingFactor = 1.0;// la place smoothing factor
    static const size_t kNumClasses = 10;                // number of classes that images_ can belong to
    static const size_t kShadeCount = 2;                 // number-coded shade. 0 being unshaded and 1 being shaded

    std::vector<Image> images_; // vector of individual training images
    std::vector<size_t> labels_;// vector of labels of training images

    std::vector<double> prior_prob_;// vector of priors at a class
    std::vector<size_t> class_;     // vector of number of images that belong to a class

    double feature_prob_[kImageSize][kImageSize][kNumClasses][kShadeCount];
    size_t feature_count_[kImageSize][kImageSize][kNumClasses][kShadeCount] = {{{{0}}}};

    // todo: make methods private
    /**
    * Method that parses images_ from data file_path.
    *
    * @param file_path of file to be parsed
    */
    void ParseImages(std::string &file_path);

    /**
     * Method that trains the model.
     */
    void TrainModel();

    /**
    * Method that calculates and sets the prior of the classes.
    */
    void CalculatePriorProbabilities();

    /**
    * Operator overload that saves data into a file.
    */
    friend std::ostream &operator<<(std::ostream &os, Model &model);

    /**
     * Method that loads data into a file.
     */
    void LoadData(std::string file);

    /**
     * Method that calculates and updates all feature probability of pixels.
     * 
     * @param row number
     * @param col number
     * @param desired_class class that image belongs to
     * @param desired_shade 0 being unshaded 1 being shaded
     */
    void CalculateFeatureProbabilities();
  };
}// namespace naivebayes

#endif//NAIVE_BAYES_MODEL_H