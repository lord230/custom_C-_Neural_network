#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

using namespace std;

const int INPUT_SIZE = 28 * 28;
const int NUM_CLASSES = 10;
const int TRAIN_SAMPLES = 60000;
const int TEST_SAMPLES = 10000;
const float LEARNING_RATE = 0.01;
const int EPOCHS = 5;

int readInt(ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read((char *)bytes, 4);
    return (int(bytes[0]) << 24) | (int(bytes[1]) << 16) | (int(bytes[2]) << 8) | int(bytes[3]);
}

vector<vector<float>> load_images(const string &filename, int &num_images) {
    ifstream file(filename, ios::binary);
    assert(file.is_open());

    int magic = readInt(file);
    num_images = readInt(file);
    int rows = readInt(file);
    int cols = readInt(file);

    vector<vector<float>> images(num_images, vector<float>(rows * cols));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images[i][j] = float(pixel) / 255.0f;
        }
    }
    return images;
}

vector<int> load_labels(const string &filename, int &num_labels) {
    ifstream file(filename, ios::binary);
    assert(file.is_open());

    int magic = readInt(file);
    num_labels = readInt(file);

    vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        labels[i] = int(label);
    }

    return labels;
}

// Softmax function
vector<float> softmax(const vector<float> &logits) {
    float maxLogit = *max_element(logits.begin(), logits.end());
    vector<float> exps(NUM_CLASSES);
    float sum = 0.0f;

    for (int i = 0; i < NUM_CLASSES; ++i) {
        exps[i] = exp(logits[i] - maxLogit); // For numerical stability
        sum += exps[i];
    }

    for (int i = 0; i < NUM_CLASSES; ++i) {
        exps[i] /= sum;
    }

    return exps;
}

// Cross-entropy loss + gradient (for 1 sample)
float cross_entropy_loss(const vector<float> &pred, int label) {
    return -log(pred[label] + 1e-8f);
}

// Model: weights + biases
struct Model {
    vector<vector<float>> W;  // shape: NUM_CLASSES x INPUT_SIZE
    vector<float> b;          // shape: NUM_CLASSES

    Model() {
        W.resize(NUM_CLASSES, vector<float>(INPUT_SIZE));
        b.resize(NUM_CLASSES);

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0.0, 0.01);

        for (int i = 0; i < NUM_CLASSES; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                W[i][j] = d(gen);
            }
            b[i] = 0.0f;
        }
    }

    vector<float> forward(const vector<float> &x) {
        vector<float> logits(NUM_CLASSES, 0.0f);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                logits[i] += W[i][j] * x[j];
            }
            logits[i] += b[i];
        }
        return softmax(logits);
    }

    void update(const vector<float> &x, const vector<float> &y_pred, int label) {
        // Gradient: dL/dz = y_pred - y_true
        for (int i = 0; i < NUM_CLASSES; ++i) {
            float error = y_pred[i] - (i == label ? 1.0f : 0.0f);
            for (int j = 0; j < INPUT_SIZE; ++j) {
                W[i][j] -= LEARNING_RATE * error * x[j];
            }
            b[i] -= LEARNING_RATE * error;
        }
    }
};

void train(Model &model,
           const vector<vector<float>> &train_images,
           const vector<int> &train_labels,
           int epochs) {
    int N = train_images.size();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_loss = 0.0;
        int correct = 0;

        for (int i = 0; i < N; ++i) {
            const vector<float> &x = train_images[i];
            int y = train_labels[i];

            vector<float> y_pred = model.forward(x);
            total_loss += cross_entropy_loss(y_pred, y);

            int pred_label = distance(y_pred.begin(), max_element(y_pred.begin(), y_pred.end()));
            if (pred_label == y) ++correct;

            model.update(x, y_pred, y);
        }

        float avg_loss = total_loss / N;
        float accuracy = 100.0f * correct / N;
        cout << "Epoch " << epoch << ": Loss = " << avg_loss << ", Accuracy = " << accuracy << "%" << endl;
    }
}

float evaluate(Model &model,
               const vector<vector<float>> &test_images,
               const vector<int> &test_labels) {
    int correct = 0;
    for (int i = 0; i < test_images.size(); ++i) {
        vector<float> pred = model.forward(test_images[i]);
        int pred_label = distance(pred.begin(), max_element(pred.begin(), pred.end()));
        if (pred_label == test_labels[i]) ++correct;
    }

    float acc = 100.0f * correct / test_images.size();
    cout << "Test Accuracy: " << acc << "%" << endl;
    return acc;
}

int main() {
    int n_train, n_test;

    cout << "Loading MNIST data..." << endl;
    auto train_images = load_images("train-images-idx3-ubyte", n_train);
    auto train_labels = load_labels("train-labels-idx1-ubyte", n_train);

    auto test_images = load_images("t10k-images-idx3-ubyte", n_test);
    auto test_labels = load_labels("t10k-labels-idx1-ubyte", n_test);

    Model model;

    cout << "Starting training..." << endl;
    train(model, train_images, train_labels, EPOCHS);

    cout << "Evaluating on test data..." << endl;
    evaluate(model, test_images, test_labels);

    return 0;
}
