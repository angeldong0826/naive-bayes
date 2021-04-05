#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <vector>
#include "core/image.h"

namespace naivebayes {

  class Model {
    /**
    * Method that parses images from data file.
    * @param file_name the file you want to parse from
    * @return vector of images
    */
    std::vector<Image> ParseImage(const std::string file);

    /**
     * Method that parses labels from data file.
     * @param file_name
     * @return vector of labels as size_t
     */
    std::vector<size_t> ParseLabel(const std::string file);
  };
}




#endif//NAIVE_BAYES_MODEL_H
