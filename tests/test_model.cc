#include <catch2/catch.hpp>

#include "core/image.h"
#include <core/model.h>
#include <fstream>

namespace naivebayes {

  TEST_CASE("Parse Image") {
    std::vector<naivebayes::Image> vector;
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimagesmall.txt";
    vector = model.ParseImage(path);
    for (auto image: vector) {
      REQUIRE(image.grid[0][0] == '0');
    }

    REQUIRE(vector[0].grid[10][11] == '1');
  }
  
  TEST_CASE("Prior probability") {
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimagesmall.txt";
    model.ParseImage(path);
    
    REQUIRE(model.CalculatePrior(4) == Approx(-2.30258));
  }
}