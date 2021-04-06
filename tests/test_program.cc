#include <catch2/catch.hpp>

#include <core/model.h>
#include "core/image.h"

namespace naivebayes {
  TEST_CASE("Check that 126 is the best class") {
    REQUIRE(naivebayes::Model().GetBestClass() == "CS 126");
  }

  TEST_CASE("Parse Image") {
    std::vector<naivebayes::Image> vector;
    vector = naivebayes::Model::ParseImage("/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimagesmall");
    for (auto image: vector) {
      REQUIRE(image.grid[0][0] == '0');
    }

    REQUIRE(vector[0].grid[10][11] == '1');
  }
}