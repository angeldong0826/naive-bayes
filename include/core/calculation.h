#ifndef NAIVE_BAYES_CALCULATION_H
#define NAIVE_BAYES_CALCULATION_H

#include <cstddef>
#include <string>

#include "core/model.h"

namespace naivebayes {

  class Calculation {

  public:
    constexpr static double kSmoothingFactor = 1.0; // la place smoothing factor
    const static size_t kNumClasses = 10; // number of classes that images can belong to

    size_t class_array[kNumClasses]; // array with size of numClasses to store frequency of image

    /**
    * Method that calculates the prior of input class.
    *
    * @param prior class to be calculated
    * @return probability of image belonging to the input prior class
    */
    double CalculatePrior(size_t input);
  };
}
#endif//NAIVE_BAYES_CALCULATION_H
