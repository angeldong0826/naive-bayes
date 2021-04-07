#include <iostream>

#include <core/model.h>
#include "core/image.h"

int main() {
  naivebayes::Model model;

  std::vector<naivebayes::Image> vector;
  vector = model.ParseImage("../data/testimagesmall.txt");
  for (auto image: vector) {
    for (auto & row : image.grid) {
      for (char i : row) {
        std::cout<< i;
      }
      std::cout << std::endl;
    }
  }
  
//  for (size_t row = 0; row < 28; ++row) {
//    for (size_t col = 0; col < 28; ++col) {
//      std::cout << vector[1].grid[row][col];
//    }
//    std::cout << std::endl;
//  }

  std::cout << "Parsing images" << std::endl;
  std::vector<naivebayes::Image> images = model.ParseImage("../data/trainingimagesandlabels.txt");


  std::cout << "Parsing labels" << std::endl;
  std::vector<size_t> labels = model.ParseLabel("../data/trainingimagesandlabels.txt");
  
  std::cout << model.CalculatePrior(4) << std::endl;
}