#include <catch2/catch.hpp>

#include "core/image.h"
#include <core/model.h>
#include <fstream>
#include <iostream>

namespace naivebayes {

  TEST_CASE("Parse Images") {
    SECTION("28 x 28") {
      Model model(28);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
      model.ParseImages(path);

      REQUIRE(model.GetImages().size() == 3);
      REQUIRE(model.GetImages()[0].GetValue(0, 0) == 0);
      REQUIRE_FALSE(model.GetImages()[0].GetValue(0, 0) == 1);
      REQUIRE(model.GetImages()[0].GetValue(10, 11) == 1);
      REQUIRE_FALSE(model.GetImages()[0].GetValue(10, 11) == 0);
      REQUIRE(model.GetImages()[1].GetValue(8, 17) == 1);
      REQUIRE_FALSE(model.GetImages()[1].GetValue(8, 17) == 0);
    }

    SECTION("5 x 5") {
      Model model(5);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
      model.ParseImages(path);

      REQUIRE(model.GetImages().size() == 10);
      REQUIRE(model.GetImages()[0].GetValue(0, 0) == 0);
      REQUIRE(model.GetImages()[0].GetValue(0, 1) == 1);
      REQUIRE(model.GetImages()[0].GetValue(0, 2) == 1);
      REQUIRE(model.GetImages()[0].GetValue(0, 3) == 1);
      REQUIRE(model.GetImages()[0].GetValue(0, 4) == 0);
      REQUIRE(model.GetImages()[0].GetValue(1, 0) == 0);
      REQUIRE(model.GetImages()[0].GetValue(1, 1) == 1);
      REQUIRE(model.GetImages()[0].GetValue(1, 2) == 0);
      REQUIRE(model.GetImages()[0].GetValue(1, 3) == 1);
      REQUIRE(model.GetImages()[0].GetValue(1, 4) == 0);
      REQUIRE(model.GetImages()[0].GetValue(2, 0) == 0);
      REQUIRE(model.GetImages()[0].GetValue(2, 1) == 0);
      REQUIRE(model.GetImages()[0].GetValue(2, 2) == 1);
      REQUIRE(model.GetImages()[0].GetValue(2, 3) == 0);
      REQUIRE(model.GetImages()[0].GetValue(2, 4) == 0);
      REQUIRE(model.GetImages()[0].GetValue(3, 0) == 0);
      REQUIRE(model.GetImages()[0].GetValue(3, 1) == 1);
      REQUIRE(model.GetImages()[0].GetValue(3, 2) == 0);
      REQUIRE(model.GetImages()[0].GetValue(3, 3) == 1);
      REQUIRE(model.GetImages()[0].GetValue(3, 4) == 0);
      REQUIRE(model.GetImages()[0].GetValue(4, 0) == 0);
      REQUIRE(model.GetImages()[0].GetValue(4, 1) == 1);
      REQUIRE(model.GetImages()[0].GetValue(4, 2) == 1);
      REQUIRE(model.GetImages()[0].GetValue(4, 3) == 1);
      REQUIRE(model.GetImages()[0].GetValue(4, 4) == 0);
    }
  }

  TEST_CASE("Parse Labels") {
    SECTION("28 x 28") {
      Model model(28);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
      model.ParseImages(path);

      REQUIRE(model.GetLabels().size() == 3);
      REQUIRE(model.GetLabels()[0] == 5);
      REQUIRE(model.GetLabels()[1] == 0);
      REQUIRE(model.GetLabels()[2] == 4);
      REQUIRE_FALSE(model.GetLabels()[0] == 1);
    }

    SECTION("5 x 5") {
      Model model(5);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
      model.ParseImages(path);

      REQUIRE(model.GetLabels().size() == 10);
      REQUIRE(model.GetLabels()[0] == 8);
      REQUIRE(model.GetLabels()[1] == 7);
      REQUIRE(model.GetLabels()[2] == 9);
      REQUIRE(model.GetLabels()[3] == 0);
      REQUIRE(model.GetLabels()[4] == 1);
      REQUIRE(model.GetLabels()[5] == 3);
      REQUIRE(model.GetLabels()[6] == 2);
      REQUIRE(model.GetLabels()[7] == 4);
      REQUIRE(model.GetLabels()[8] == 5);
      REQUIRE(model.GetLabels()[9] == 6);
    }
  }

  TEST_CASE("Prior probabilities") {
    SECTION("28 x 28") {
      Model model(28);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
      model.ParseImages(path);

      model.CalculatePriorProbabilities();

      REQUIRE(model.prior_prob_[0] == Approx(-1.8718021769));
      REQUIRE(model.prior_prob_[1] == Approx(-2.5649493575));
      REQUIRE(model.prior_prob_[2] == Approx(-2.5649493575));
      REQUIRE(model.prior_prob_[3] == Approx(-2.5649493575));
      REQUIRE(model.prior_prob_[4] == Approx(-1.8718021769));
      REQUIRE(model.prior_prob_[5] == Approx(-1.8718021769));
      REQUIRE(model.prior_prob_[6] == Approx(-2.5649493575));
      REQUIRE(model.prior_prob_[7] == Approx(-2.5649493575));
      REQUIRE(model.prior_prob_[8] == Approx(-2.5649493575));
      REQUIRE(model.prior_prob_[9] == Approx(-2.5649493575));

      // makes sense because the three images belong to class 0, 4, and 5
    }

    SECTION("5 x 5") {
      Model model(5);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
      model.ParseImages(path);

      model.CalculatePriorProbabilities();

      REQUIRE(model.prior_prob_[0] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[1] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[2] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[3] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[4] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[5] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[6] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[7] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[8] == Approx(-2.302585093));
      REQUIRE(model.prior_prob_[9] == Approx(-2.302585093));

      // makes sense because there is an image for every class
    }
  }

  TEST_CASE("Feature probability") {
    SECTION("28 x 28") {
      Model model(28);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/smalltestimage.txt";
      model.ParseImages(path);

      model.TrainModel();// train model calculates and updates feature_prob_

      REQUIRE(model.feature_prob_[0][0][0][0] == Approx(-0.4054651081));
      REQUIRE(model.feature_prob_[0][1][2][0] == Approx(-0.6931471806));
    }

    SECTION("5 x 5") {
      Model model(5);
      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
      model.ParseImages(path);

      model.TrainModel();

      REQUIRE(model.feature_prob_[0][0][4][0] == Approx(-1.0986122887));
      REQUIRE(model.feature_prob_[1][0][2][0] == Approx(-0.4054651081));
    }
  }

  TEST_CASE("Operator overload") {
    SECTION("Save Data <<") {
      Model model(5);

      std::string file = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
      model.ParseImages(file);
      model.TrainModel();

      std::string file_path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ofstream output(file_path);

      if (output.is_open()) {
        output << model;
      }

      REQUIRE(output.is_open());
      output.close();

      std::string expected_file = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/expectedsavefile.txt";
      std::ifstream read;
      read.open(expected_file);
      
      std::string a;
      std::string expected;
      while (read >> expected) {
        a += expected;
      }

      std::string actual_file = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/expectedsavefile.txt";
      std::ifstream read_actual;
      read_actual.open(actual_file);
      
      std::string b;
      std::string actual;
      while (read_actual >> actual) {
        b += actual;
      }

      REQUIRE(a == b);
    }

    SECTION("Load Data >>") {
      Model model(5);

      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      REQUIRE(my_file.is_open());
      my_file.close();
    }
  }

}// namespace naivebayes
