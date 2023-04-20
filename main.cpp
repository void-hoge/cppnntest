#include "activation.hpp"
#include "layer.hpp"
#include "model.hpp"

#include "cppmnist.hpp"

#include <iostream>

int main() {
	auto mnist_train_images = MNIST::Images<double>("../mnist/train-images-idx3-ubyte");
	auto mnist_train_labels = MNIST::Labels("../mnist/train-labels-idx1-ubyte");
	std::cout << "train files loaded" << std::endl;
	auto num = mnist_train_labels.data().size();
	auto height = mnist_train_images.data()[0].size();
	auto width = mnist_train_images.data()[0][0].size();
	auto train_x = std::vector<std::vector<double>>(num, std::vector<double>(width*height));
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				train_x[i][j*width+k] = mnist_train_images.data()[i][j][k];
			}
		}
	}
	auto train_y = mnist_train_labels.onehot();
	auto batchsize = 30;

	Model model(width*height);
	model.add_dense_layer(30, ActivationType::relu);
	model.add_dense_layer(30, ActivationType::relu);
	model.add_dense_layer(10, ActivationType::softmax);
	auto history = model.fitbp(num/batchsize, 0.1, train_x, train_y, batchsize, "mse");

	auto mnist_test_images = MNIST::Images<double>("../mnist/t10k-images-idx3-ubyte");
	auto mnist_test_labels = MNIST::Labels("../mnist/t10k-labels-idx1-ubyte");
	std::cout << "test files loaded" << std::endl;
	auto testnum = mnist_test_labels.data().size();
	auto test_x = std::vector<std::vector<double>>(num, std::vector<double>(width*height));
	for (int i = 0; i < testnum; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				test_x[i][j*width+k] = mnist_test_images.data()[i][j][k];
			}
		}
	}
	int count = 0;
	auto predict_y = model.predict(test_x);
	for (int i = 0; i < testnum; i++) {
		int maxidx = 0;
		double maxval = -1000;
		for (int j = 0; j < 10; j++) {
			if (predict_y[i][j] > maxval) {
				maxidx = j;
				maxval = predict_y[i][j];
			}
		}
		if (maxidx == mnist_test_labels.data()[i]) {
			count++;
		}else {
			mnist_test_images.dump(i, std::cout);
			std::cout << "index: " << i << ", predict: " << maxidx << ", correct: " << mnist_test_labels.data()[i] << std::endl;
		}
	}
	std::cout << count << "/" << testnum << " (" << (double)count*100/testnum << " %)" << std::endl;
	return 0;
}
