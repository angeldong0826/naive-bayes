#include <catch2/catch.hpp>

#include "core/image.h"
#include <core/model.h>
#include <fstream>

namespace naivebayes {

  TEST_CASE("Parse Image") {
    std::vector<naivebayes::Image> vector;
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimage.txt";
    vector = model.ParseImages(path);

    REQUIRE(vector.size() == 3);
    REQUIRE(vector[0].grid[0][0] == '0');
    REQUIRE(vector[0].grid[10][11] == '1');
    REQUIRE(vector[1].grid[8][17] == '1');
    REQUIRE(vector[2].grid[0][0] == '0');
    REQUIRE_FALSE(vector[0].grid[0][0] == '5');
  }
  
  TEST_CASE("Parse Label") {
    std::vector<size_t> vector;
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimage.txt";
    vector = model.ParseLabel(path);
     
     REQUIRE(vector.size() == 3);
     REQUIRE(vector[0] == 5);
     REQUIRE(vector[1] == 0);
     REQUIRE(vector[2] == 4);
     REQUIRE_FALSE(vector[0] == 1);
  }
  
  TEST_CASE("Prior probability") {
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimage.txt";
    
    REQUIRE(model.CalculatePrior(4, path) == Approx(-1.8718021769));
    REQUIRE(model.CalculatePrior(9, path) == Approx(-2.5649493575));
  }
  
  TEST_CASE("Pixel probability") {
    Model model;

    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimage.txt";
    model.ParseImages(path);
    model.ParseLabel(path);

    REQUIRE(model.CalculateFeatureProbability(0, 0, 0, 0) == Approx(-0.6931471));
    REQUIRE(model.CalculateFeatureProbability(12, 12, 0, 3) == Approx(-0.6931471));
  }
  
  TEST_CASE("Save data") {
    std::string file_path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/emptyfiletoload.txt";
    
    std::ofstream output(file_path);
    
    Model model;
    
    if (output.is_open()) {
      output << model;
    }
    
    REQUIRE(output.is_open());
    output.close();
  }
  
  TEST_CASE("Load data") {
    naivebayes::Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/loadfile.txt";
    model.LoadData(path, model);
    REQUIRE(model.CalculatePrior(0, path) == Approx(-0.1805083197));
  }
}

// check PL
