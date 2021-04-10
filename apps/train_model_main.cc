#include <iostream>

#include "core/image.h"
#include <core/model.h>
#include <fstream>

int main() {
  naivebayes::Model model;
//  std::vector<naivebayes::Image> vector;
//  vector = model.ParseImages("../data/testimage.txt");

//  for (auto image: vector) {
//    for (auto & row : image.grid) {
//      for (char i : row) {
//        std::cout<< i;
//      }
//      std::cout << std::endl;
//    }
//  }
  
  std::string path = "../data/trainingimagesandlabels.txt";
  
  std::cout << "Parsing images" << std::endl;
  model.ParseImages(path);
  
//  std::cout << "Parsing labels" << std::endl;
//  model.ParseLabel(path);
//  model.Train();
  std::cout << model.labels_.size() << std::endl;
  std::cout << model.images_.size() << std::endl;

//  std::cout << model.CalculatePrior(4, path) << std::endl;

  std::string file_path = "../data/emptyfiletoloadmain.txt";
  
  std::string new_file_path = "../data/modeltoload.txt";

  // std::ofstream output(file_path);
  std::ofstream output(new_file_path);

  if (output.is_open()) {
    output << model;
    output.close();
  }
}