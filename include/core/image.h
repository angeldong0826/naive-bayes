#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#include <cstddef>
#include <string>
#include <vector>

namespace naivebayes {

  const size_t kImageSize = 28; // size of each image to be processed
  const std::string file = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/trainingimagesandlabels.txt";


  /**
   * Class that encapsulates training images.
   */
  class Image {

  public:

    std::vector<Image> images; // array storing the image pixels

    char grid[kImageSize][kImageSize]; // array storing the image pixels

    /**
     * Overloading operator >> to convert an image to a vector of 0's and 1's.
     *
     * @param is input stream
     * @param image
     * @return vector of Image of 0's and 1's corresponding to color
     */
    friend std::istream& operator>>(std::istream& is, Image& image);
  };

}// namespace naivebayes

#endif//NAIVE_BAYES_IMAGE_H
