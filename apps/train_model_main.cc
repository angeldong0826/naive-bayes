#include <iostream>

#include "core/image.h"
#include <core/model.h>
#include <fstream>

int main() {
  naivebayes::Model model;
  
  std::string path = "../data/trainingimagesandlabels.txt";
  
  model.ParseImages(path);
//  for (int i = 0; i < 28; i++) {
//    for (int j = 0; j < 28; ++j) {
//      std::cout << model.images_[0].grid[i][j];
//    }
//    std::cout << std::endl;
//  }

  model.TrainModel();
  
  std::string file_path = "../data/modeltoload.txt";

   std::ofstream output(file_path);

  if (output.is_open()) {
    output << model;
    output.close();
  }
}