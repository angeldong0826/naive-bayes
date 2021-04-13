#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace naivebayes {

  const size_t kImageSize = 28;// size of each image_ to be processed

  /**
   * Class that encapsulates training images.
   */
  class Image {

  public:
    /**
     * Operator overload that converts an image_ to an array of processed 0's and 1's.
     *
     * @param is input stream
     * @param image
     * @return array of Image of 0's and 1's corresponding to shade
     */
    friend std::istream &operator>>(std::istream &is, Image &image);

    /**
     * Getter method that gets the value at index of image_ grid.
     * 
     * @param row number
     * @param column number
     * @return value at index
     */
    char GetValue(size_t row, size_t col) const;
    
    void SetValue(size_t row, size_t col, size_t value);
    
  private:
    char grid[kImageSize][kImageSize];// array storing the image_ pixels
  };

}// namespace naivebayes
