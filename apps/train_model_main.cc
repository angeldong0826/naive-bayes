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
  
  naivebayes::Classifier classifier(model);

  std::cout << classifier.CalculateAccuracyPercentage(model) << "%" << std::endl;
  
//  std::string file_path = "../data/modeltoload.txt";
//
//  std::ofstream output(file_path);
//
//  if (output.is_open()) {
//    output << model;
//    output.close();
//  }
  
}