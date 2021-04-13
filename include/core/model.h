#pragma once

#include "core/image.h"
#include "ostream"

namespace naivebayes {

  const constexpr double kSmoothingFactor = 1.0;// la place smoothing factor
  const size_t kNumClasses = 10;                // number of classes that images_ can belong to
  const size_t kShadeCount = 2;                 // number-coded shade. 0 being unshaded and 1 being shaded
  
  /**
   * Class that models image_ training.
   */
  class Model {
  public:
    std::vector<double> prior_prob_;// vector of priors at a class
    std::vector<size_t> class_;     // vector of number of images that belong to a class

    std::vector<std::vector<std::vector<std::vector<double>>>> feature_prob_;
    std::vector<std::vector<std::vector<std::vector<size_t>>>> feature_count_;

//    double feature_prob_[kImageSize][kImageSize][kNumClasses][kShadeCount] = {{{{0}}}};
//    size_t feature_count_[kImageSize][kImageSize][kNumClasses][kShadeCount] = {{{{0}}}};
    
    Model(size_t size);

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
    void LoadData(std::string &file);

    /**
     * Method that calculates and updates all feature probability of pixels.
     * 
     * @param row number
     * @param col number
     * @param desired_class class that image_ belongs to
     * @param desired_shade 0 being unshaded 1 being shaded
     */
    void CalculateFeatureProbabilities();

    /**
     * Getter method to get images.
     * 
     * @return images vector
     */
    std::vector<Image> GetImages();
    
    /**
     * Getter method to get labels.
     * 
     * @return labels vector
     */
    std::vector<size_t> GetLabels();
    
  private:
    std::vector<Image> images_; // vector of individual training images
    std::vector<size_t> labels_;// vector of labels of training images
  };
}// namespace naivebayes
