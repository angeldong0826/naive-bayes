#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#include <cstddef>
#include <vector>

namespace naivebayes {

  const size_t kImageSize = 28; // size of each image to be processed

  /**
   * Class that encapsulates training images.
   */
  class Image {

  public:
    /**
     * Array storing the image pixels.
     */
    char grid[kImageSize][kImageSize];

    /**
     * Method to get rid of gray pixels and makes image white and black, white being 0 and black being 1.
     */
    void NumberCodeImages();

    /**
     *
     *
     * @param is
     * @param image
     * @return vector of Image of 0's and 1's corresponding to color
     */
    friend std::istream& operator>>(std::istream& is, Image& image);

    friend std::ostream& operator<<(std::ostream& os, Image& image);
  };

}// namespace naivebayes

#endif//NAIVE_BAYES_IMAGE_H
