#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <vector>
#include "core/image.h"

namespace naivebayes {

  class Model {
  public:
    /**
    * Method that parses images from data file_path.
     *
    * @param file_path of file to be parsed
    */
    static void ParseImage(std::string file_path);

    /**
     * Method that parses labels from data file_path.
     *
     * @param file_path of file to be parsed
     */
    void ParseLabel(std::string file_path);

    static std::vector<Image> images;

    static std::vector<size_t> labels;

    std::string GetBestClass() const;
  };
}




#endif//NAIVE_BAYES_MODEL_H
