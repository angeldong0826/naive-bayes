#include <catch2/catch.hpp>

#include "core/image.h"
#include <core/model.h>
#include <fstream>

namespace naivebayes {

  TEST_CASE("Parse Image") {
    std::vector<Image> vector;
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
    model.ParseImages(path);

    REQUIRE(model.GetImages().size() == 3);
    REQUIRE(model.GetImages()[0].GetValue(0, 0) == '0');
    REQUIRE_FALSE(model.GetImages()[0].GetValue(0, 0) == '1');
    REQUIRE(model.GetImages()[0].GetValue(10, 11) == '1');
    REQUIRE_FALSE(model.GetImages()[0].GetValue(10, 11) == '0');
    REQUIRE(model.GetImages()[1].GetValue(8, 17) == '1');
    REQUIRE_FALSE(model.GetImages()[1].GetValue(8, 17) == '0');
  }

  TEST_CASE("Parse Label") {
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
    model.ParseImages(path);

    REQUIRE(model.GetLabels().size() == 3);
    REQUIRE(model.GetLabels()[0] == 5);
    REQUIRE(model.GetLabels()[1] == 0);
    REQUIRE(model.GetLabels()[2] == 4);
    REQUIRE_FALSE(model.GetLabels()[0] == 1);
  }

  TEST_CASE("Prior probability") {
    Model model;
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";

    model.CalculatePriorProbabilities();

    REQUIRE(model.prior_prob_[0] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[1] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[2] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[3] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[4] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[5] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[6] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[7] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[8] == Approx(-2.302585));
    REQUIRE(model.prior_prob_[9] == Approx(-2.302585));

  }
  //
  //  TEST_CASE("Pixel probability") {
  //    Model model;
  //
  //    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
  //    model.ParseImages(path);
  //    model.ParseLabel(path);
  //
  //    REQUIRE(model.CalculateFeatureProbabilities() == Approx(-0.6931471));
  //    REQUIRE(model.CalculateFeatureProbabilities() == Approx(-0.6931471));
  //  }

  TEST_CASE("Save data") {
    std::string file_path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testsavedata.txt";

    std::ofstream output(file_path);

    Model model;
    model.TrainModel();

    if (output.is_open()) {
      output << model;
    }

    REQUIRE(output.is_open());
    output.close();
  }

  //  TEST_CASE("Load data") {
  //    naivebayes::Model model;
  //    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/loadfile.txt";
  //    model.LoadData(path, model);
  //    REQUIRE(model.CalculatePriorProbabilities(0, path) == Approx(-0.1805083197));
  //  }
}// namespace naivebayes

// todo: check PL
