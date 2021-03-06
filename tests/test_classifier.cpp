#include "core/classifier.h"
#include "core/image.h"
#include "core/model.h"
#include <catch2/catch.hpp>
#include <fstream>

namespace naivebayes {

  TEST_CASE("Likelihood score") {
    Model model(5);
    std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
    std::ifstream my_file(p);
    my_file >> model;

    Image image(5);
    Classifier classifier(model);

    SECTION("Zero") {
      std::vector<std::vector<size_t>> vector = {{0, 0, 0, 0, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 1, 0, 1, 0},
                                                 {0, 1, 1, 1, 0},
                                                 {0, 0, 0, 0, 0}};
      image.SetGridVector(vector);
      classifier.ReturnPredictedClass(image);

      REQUIRE(classifier.GetLikelihoodScore()[0] == Approx(-12.439215));// highest likelihood
      REQUIRE(classifier.GetLikelihoodScore()[1] == Approx(-17.29123));
      REQUIRE(classifier.GetLikelihoodScore()[2] == Approx(-20.06381));
      REQUIRE(classifier.GetLikelihoodScore()[3] == Approx(-20.06381));
      REQUIRE(classifier.GetLikelihoodScore()[4] == Approx(-20.06381));
      REQUIRE(classifier.GetLikelihoodScore()[5] == Approx(-26.99526));
      REQUIRE(classifier.GetLikelihoodScore()[6] == Approx(-27.688405));
      REQUIRE(classifier.GetLikelihoodScore()[7] == Approx(-19.370665));
      REQUIRE(classifier.GetLikelihoodScore()[8] == Approx(-20.06381));
      REQUIRE(classifier.GetLikelihoodScore()[9] == Approx(-17.984375));
    }

    SECTION("One") {
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

  TEST_CASE("Predicted Class") {
    Model model(5);
    std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoloadfortest.txt";
    std::ifstream my_file(p);
    my_file >> model;

    SECTION("Zero") {
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
    SECTION("<70% test") {
      naivebayes::Model model(28);

      std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/testimagesandlabels.txt";

      model.ParseImages(path);
      model.TrainModel();

      std::string p = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/modeltoload.txt";
      std::ifstream my_file(p);
      my_file >> model;

      naivebayes::Classifier classifier(model);

      REQUIRE(classifier.CalculateAccuracyPercentage(model) == Approx(77.1));
    }

    SECTION("100% custom test file") {
      Model model(5);

      std::string file = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";

      model.ParseImages(file);
      model.TrainModel();

      Classifier classifier(model);

      REQUIRE(classifier.CalculateAccuracyPercentage(model) == 100.0);
    }
  }

  TEST_CASE("Confusion Matrix") {
    naivebayes::Model model(5);
    std::string path = "/Users/angeldong/CLionProjects/Cinder/my-projects/naive-bayes-angeldong0826/data/customtestimages.txt";
    model.ParseImages(path);
    model.TrainModel();

    REQUIRE(model.GetImages().size() == 10);
    REQUIRE(model.GetLabels().size() == 10);

    naivebayes::Classifier classifier(model);

    std::vector<std::vector<size_t>> matrix = classifier.GetConfusionMatrix(model.GetImages(), model.GetLabels());

    // confusion matrix:
    // 1 0 0 0 0 0 0 0 0 0
    // 0 1 0 0 0 0 0 0 0 0
    // 0 0 1 0 0 0 0 0 0 0
    // 0 0 0 1 0 0 0 0 0 0
    // 0 0 0 0 1 0 0 0 0 0
    // 0 0 0 0 0 1 0 0 0 0
    // 0 0 0 0 0 0 1 0 0 0
    // 0 0 0 0 0 0 0 1 0 0
    // 0 0 0 0 0 0 0 0 1 0
    // 0 0 0 0 0 0 0 0 0 1

    // should be completely accurate - 1's on the diagonal

    REQUIRE(matrix[0][0] == 1);
    REQUIRE(matrix[0][1] == 0);
    REQUIRE(matrix[0][9] == 0);
    REQUIRE(matrix[1][1] == 1);
    REQUIRE(matrix[2][2] == 1);
    REQUIRE(matrix[3][3] == 1);
    REQUIRE(matrix[4][4] == 1);
    REQUIRE(matrix[5][5] == 1);
    REQUIRE(matrix[6][6] == 1);
    REQUIRE(matrix[7][7] == 1);
    REQUIRE(matrix[8][8] == 1);
    REQUIRE(matrix[9][9] == 1);
    REQUIRE(matrix[9][8] == 0);
  }

}// namespace naivebayes