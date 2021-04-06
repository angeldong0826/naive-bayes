#include <catch2/catch.hpp>

#include <core/model.h>
#include "core/image.h"

namespace naivebayes {

  TEST_CASE("Parse Image") {
    std::vector<naivebayes::Image> vector;
    vector = naivebayes::Model::ParseImage("../data/testimagesmall");
    for (auto image: vector) {
      REQUIRE(image.grid[0][0] == '0');
    }

    REQUIRE(vector[0].grid[10][11] == '1');
  }
}