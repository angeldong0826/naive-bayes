#include <iostream>

#include <core/model.h>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.

  std::cout << "Welcome to " << naivebayes::Model().GetBestClass()
            << std::endl;
  return 0;
}
