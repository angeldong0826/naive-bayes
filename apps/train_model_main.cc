#include <iostream>

#include <core/model.h>

int main() {

  std::vector<naivebayes::Image> vector;
  naivebayes::Model model;
  vector = model.ParseImage("../data/trainingimagesandlabels.txt");
  for (auto image: vector) {
    for (auto & row : image.grid) {
      for (char i : row) {
        std::cout<< i;
      }
      std::cout << std::endl;
    }
  }
  
  
}