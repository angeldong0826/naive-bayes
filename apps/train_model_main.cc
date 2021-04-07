#include <iostream>

#include "core/image.h"
#include <core/model.h>
#include <fstream>

int main() {
  naivebayes::Model model;
//  std::vector<naivebayes::Image> vector;
//  vector = model.ParseImage("../data/testimage.txt");
//
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
  model.ParseImage(path);
  
  std::cout << "Parsing labels" << std::endl;
  model.ParseLabel(path);
  
  std::cout << model.CalculatePrior(4, path) << std::endl;

  std::string file_path = "../data/emptyfiletoloadmain.txt";

  std::ofstream output(file_path);

  if (output.is_open()) {
    output << model;
    output.close();
  }
}