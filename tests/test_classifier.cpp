#include "core/classifier.h"
#include "core/image.h"
#include "core/model.h"
#include <catch2/catch.hpp>
#include <fstream>

namespace naivebayes {

  TEST_CASE("Likelihood score") {
    SECTION("one") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 1, 1, 0, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 0, 0, 0, 0}};
      image.SetGridVector(vector);
      classifier.ReturnPredictedClass(image);

      REQUIRE(classifier.GetLikelihoodScore()[0] == Approx(-17.29123));
      REQUIRE(classifier.GetLikelihoodScore()[1] == Approx(-12.439215));// highest likelihood
      REQUIRE(classifier.GetLikelihoodScore()[2] == Approx(-19.370665));
      REQUIRE(classifier.GetLikelihoodScore()[3] == Approx(-16.598085));
      REQUIRE(classifier.GetLikelihoodScore()[4] == Approx(-19.370665));
      REQUIRE(classifier.GetLikelihoodScore()[5] == Approx(-24.915825));
      REQUIRE(classifier.GetLikelihoodScore()[6] == Approx(-25.60897));
      REQUIRE(classifier.GetLikelihoodScore()[7] == Approx(-15.90494));
      REQUIRE(classifier.GetLikelihoodScore()[8] == Approx(-17.984375));
      REQUIRE(classifier.GetLikelihoodScore()[9] == Approx(-18.67752));
    }

    SECTION("Five") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{1, 1, 1, 1, 1},
                                                 {1, 0, 0, 0, 0},
                                                 {1, 1, 1, 1, 1},
                                                 {0, 0, 0, 0, 1},
                                                 {1, 1, 1, 1, 1}};

      image.SetGridVector(vector);
      classifier.ReturnPredictedClass(image);

      REQUIRE(classifier.GetLikelihoodScore()[0] == Approx(-26.99526));
      REQUIRE(classifier.GetLikelihoodScore()[1] == Approx(-24.915825));
      REQUIRE(classifier.GetLikelihoodScore()[2] == Approx(-22.143245));
      REQUIRE(classifier.GetLikelihoodScore()[3] == Approx(-22.143245));
      REQUIRE(classifier.GetLikelihoodScore()[4] == Approx(-19.370665));
      REQUIRE(classifier.GetLikelihoodScore()[5] == Approx(-12.439215));// highest likelihood
      REQUIRE(classifier.GetLikelihoodScore()[6] == Approx(-13.13236));
      REQUIRE(classifier.GetLikelihoodScore()[7] == Approx(-22.83639));
      REQUIRE(classifier.GetLikelihoodScore()[8] == Approx(-22.143245));
      REQUIRE(classifier.GetLikelihoodScore()[9] == Approx(-21.4501));
    }
  }

  TEST_CASE("Predict Class") {
    SECTION("Zero") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 0, 0, 0, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 0, 0, 0, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 0);
    }
    
    SECTION("One") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 1, 1, 0, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 0, 0, 0, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 1);
    }

    SECTION("Two") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 0, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 0, 0, 1, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 1, 1, 1, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 2);
    }

    SECTION("Three") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 1, 1, 0, 0},
                                                 {0, 0, 0, 1, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 0, 0, 1, 0},
                                                 {0, 1, 1, 0, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 3);
    }

    SECTION("Four") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{1, 0, 1, 0, 0},
                                                 {1, 0, 1, 0, 0},
                                                 {1, 1, 1, 1, 1},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 0, 1, 0, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 4);
    }

    SECTION("Five") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{1, 1, 1, 1, 1},
                                                 {1, 0, 0, 0, 0},
                                                 {1, 1, 1, 1, 1},
                                                 {0, 0, 0, 0, 1},
                                                 {1, 1, 1, 1, 1}};

      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 5);
    }

    SECTION("Six") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{1, 1, 1, 1, 1},
                                                 {1, 0, 0, 0, 0},
                                                 {1, 1, 1, 1, 1},
                                                 {1, 0, 0, 0, 1},
                                                 {1, 1, 1, 1, 1}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 6);
    }

    SECTION("Seven") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 1, 1, 1, 0},
                                                 {0, 0, 0, 1, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 0, 1, 0, 0},
                                                 {0, 0, 0, 0, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 7);
    }

    SECTION("Eight") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 1, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 1, 1, 1, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 8);
    }

    SECTION("Nine") {
      Model model(5);
      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
      std::ifstream my_file(p);
      my_file >> model;

      Image image(5);
      Classifier classifier(model);

      std::vector<std::vector<size_t>> vector = {{0, 1, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 0, 0, 1, 0}};
      image.SetGridVector(vector);
      REQUIRE(classifier.ReturnPredictedClass(image) == 9);
    }
  }

  TEST_CASE("Accuracy Percentage") {
    Model model(5);

    std::string file = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
    model.ParseImages(file);
    model.TrainModel();

    Classifier classifier(model);
    REQUIRE(classifier.CalculateAccuracyPercentage(model) == 100.0);
  }
  
}// namespace naivebayes