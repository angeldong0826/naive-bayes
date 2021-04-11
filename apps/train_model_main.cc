#include <iostream>

#include "core/image.h"
#include <core/model.h>
#include "core/classifier.h"
#include <fstream>

int main() {
  naivebayes::Model model;
  naivebayes::Classifier classifier;
  
  std::string path = "../data/trainingimagesandlabels.txt";
  
  model.ParseImages(path);
  
//  for (int i = 0; i < 28; i++) {
//    for (int j = 0; j < 28; ++j) {
//      std::cout << model.images_[0].grid[i][j];
//    }
//    std::cout << std::endl;
//  }

  model.TrainModel();

//  classifier.ReturnPredictedClass(model.images_[0], model);
  for (size_t i = 0; i < 10; ++i) {
    std::cout << classifier.ReturnPredictedClass(model.images_[i], model) << std::endl;
  }
//  std::cout << classifier.ReturnPredictedClass(model.images_[0], model) << std::endl;
//  std::cout << classifier.ReturnPredictedClass(model.images_[1], model) << std::endl;
//  std::cout << classifier.ReturnPredictedClass(model.images_[2], model) << std::endl;
//  std::cout << classifier.ReturnPredictedClass(model.images_[3], model) << std::endl;
  
  std::string file_path = "../data/modeltoload.txt";

  std::ofstream output(file_path);

  if (output.is_open()) {
    output << model;
    output.close();
  }
}