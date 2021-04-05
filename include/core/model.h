#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <vector>
#include "core/image.h"

namespace naivebayes {

  class Model {
  public:
    /**
    * Method that parses images from data file_to_parse.
    * @param file_name the file_to_parse you want to parse from
    * @return vector of images
    */
    static std::vector<Image> ParseImage(std::string file_to_parse);

    /**
     * Method that parses labels from data file_to_parse.
     * @param file_name
     * @return vector of labels as size_t
     */
    static std::vector<size_t> ParseLabel(std::string file_to_parse);
  };
}




#endif//NAIVE_BAYES_MODEL_H
