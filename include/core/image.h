#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#include <cstddef>
#include <string>
#include <vector>

namespace naivebayes {

  const size_t kImageSize = 28;// size of each image to be processed

  /**
   * Class that encapsulates training images.
   */
  class Image {

  public:
    char grid[kImageSize][kImageSize];// array storing the image pixels

    /**
     * Operator overload that converts an image to an array of processed 0's and 1's.
     *
     * @param is input stream
     * @param image
     * @return array of Image of 0's and 1's corresponding to shade
     */
    friend std::istream &operator>>(std::istream &is, Image &image);
  };

}// namespace naivebayes

#endif//NAIVE_BAYES_IMAGE_H
