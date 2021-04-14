#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace naivebayes {

  /**
   * Class that encapsulates training images.
   */
  class Image {
  public:
    Image();
    /**
     * Image constructor that sets the image size.
     * 
     * @param image_size to be set
     */
    Image(size_t image_size);
    
    /**
     * Operator overload that converts an image_ to an array of processed 0's and 1's.
     *
     * @param is input stream
     * @param image
     * @return array of Image of 0's and 1's corresponding to shade
     */
    friend std::istream &operator>>(std::istream &is, Image &image);

    /**
     * Getter method that gets the value at index of image grid.
     * 
     * @param row number
     * @param column number
     * @return value at index
     */
    size_t GetValue(size_t row, size_t col) const;

    /**
     * Setter method to set value to index of image.
     * 
     * @param row 
     * @param col 
     * @param value 
     */
    void SetValue(size_t row, size_t col, size_t value);

    void SetImageSize(size_t size);
    
    void SetGridSize(size_t size);
    
  private:
    std::vector<std::vector<size_t>> grid_;// vector storing the image pixels
    size_t image_size_;// size of image
  };

}// namespace naivebayes
