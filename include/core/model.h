#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <vector>
#include "core/image.h"

namespace naivebayes {

  /**
   * //TODO JAVADOC
   */
  class Model {
  public:
    static constexpr double kSmoothingFactor = 1.0; // la place smoothing factor
    static const size_t kNumClasses = 10; // number of classes that images_ can belong to
    const size_t kShadeCount = 2; // number-coded shade. 0 being unshaded and 1 being shaded

    static size_t class_array[kNumClasses]; // array with size of numClasses to store frequency of image

    static std::vector<Image> images_;
    static std::vector<size_t> labels_;
    
    /**
    * Method that parses images_ from data file_path.
     *
    * @param file_path of file to be parsed
    */
    static std::vector<Image> ParseImage(std::string file_path);

    /**
     * Method that parses labels_ from data file_path.
     *
     * @param file_path of file to be parsed
     */
    static std::vector<size_t> ParseLabel(std::string file_path);

    /**
     * Method that calculates the amount of training images_ that belong to a class.
     * 
     * @param row row number
     * @param col column number
     * @param desired_class class number
     * @return number of images_ that belong to input class
     */
    double CalculateShadeProbability(size_t row, size_t col, size_t desired_class, size_t desired_shade);
    
    /**
     * Method that calculates the pixel probability for every pixel in an image.
     */
    void CalculatePixelProbability();

    /**
    * Method that calculates the prior of input class.
    *
    * @param prior class to be calculated
    * @return probability of image belonging to the input prior class
    */
    double CalculatePrior(size_t input);

    //std::string GetBestClass() const;
  };
}

#endif//NAIVE_BAYES_MODEL_H