#include <iostream>

#include "core/image.h"
#include <core/model.h>
#include "core/classifier.h"
#include <fstream>

int main() {
  
  // ------calculates prediction accuracy----------
  naivebayes::Model model(28);

  std::string path = "../data/testimagesandlabels.txt";

  model.ParseImages(path);

  model.TrainModel();

  std::string p = "../data/modeltoload.txt";
  std::ifstream my_file(p);
  my_file >> model;

  naivebayes::Classifier classifier(model);

  std::cout << classifier.CalculateAccuracyPercentage(model) << "%" << std::endl;

// ------prints out confusion matrix----------

//  naivebayes::Model model(5);
//  std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
//  model.ParseImages(path);
//  model.TrainModel();
//
//  naivebayes::Classifier classifier(model);
//
//  std::vector<std::vector<size_t>> matrix = classifier.GetConfusionMatrix(model.GetImages(), model.GetLabels());
//
//  for (int i = 0; i < model.GetImages().size(); i++) {
//    for (int j = 0; j < model.GetImages().size(); j++) {
//      std::cout << classifier.GetConfusionMatrix(model.GetImages(), model.GetLabels())[i][j] << " ";
//    }
//    std::cout << std::endl;
//  }
  
//  ------saves data----------
  
//  std::string file_path = "../data/modeltoload.txt";
//
//  std::ofstream output(file_path);
//
//  if (output.is_open()) {
//    output << model;
//    output.close();
//  }
  
}