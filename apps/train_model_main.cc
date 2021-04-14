#include <iostream>

#include "core/image.h"
#include <core/model.h>
#include "core/classifier.h"
#include <fstream>

int main() {
  naivebayes::Model model(28);
  
  std::string path = "../data/testimagesandlabels.txt";
  
  model.ParseImages(path);

//  model.TrainModel();
  
  std::string p = "../data/modeltoload.txt";
  std::ifstream my_file(p);
  my_file >> model;
  
//  std::cout << model.feature_prob_[0][0][0][0];
  
  naivebayes::Classifier classifier(model);

  std::cout << classifier.CalculateAccuracyPercentage(model) << "%" << std::endl;
  
//  std::cout << classifier.ReturnPredictedClass(model.images_[0], model) << std::endl;
//  std::cout << classifier.ReturnPredictedClass(model.images_[1], model) << std::endl;
//  std::cout << classifier.ReturnPredictedClass(model.images_[2], model) << std::endl;
  
//  std::string file_path = "../data/modeltoload.txt";
//
//  std::ofstream output(file_path);
//
//  if (output.is_open()) {
//    output << model;
//    output.close();
//  }
  
}