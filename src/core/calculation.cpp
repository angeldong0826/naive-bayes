#include "core/calculation.h"

namespace naivebayes {

  double Calculation::CalculatePrior(size_t input) {
    std::vector<size_t> vector = Model::ParseLabel(file);

    for (size_t i = 0; i < vector.size(); i++) {
      if (vector[i] == input) {
        class_array[i] += 1;
      }
    }

    return (kSmoothingFactor + class_array[input]) / (kNumClasses + kSmoothingFactor * vector.size());
  }
}

