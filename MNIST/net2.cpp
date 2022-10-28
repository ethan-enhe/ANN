#include "../include/net2.h"

#include <bits/stdc++.h>

#include <cstdlib>
#include <memory>

#include "../include/optimize.h"
using namespace std;

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, vector<double>& labels) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;

        for (int i = 0; i < number_of_images; i++) {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((double)label);
        }
    }
}

void read_Mnist_Images(string filename, vector<vector<double>>& images) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;

        for (int i = 0; i < number_of_images; i++) {
            vector<double> tp;
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    tp.push_back(image);
                }
            }
            images.push_back(tp);
        }
    }
}

vector<double> pool(const vector<double>& x) {
    vector<double> res;
    for (int k = 0; k < 28; k += 2)
        for (int j = 0; j < 28; j += 2)
            res.push_back(max({x[k * 28 + j], x[k * 28 + j + 1], x[(k + 1) * 28 + j], x[(k + 1) * 28 + j + 1]}) / 255);
    return res;
}

layer_seq net;
data_set data;
int main() {
    cout << Eigen::nbThreads() << endl;
    /* mat x(2, 2); */
    /* x << 1, 2, 3, 4; */
    /* x=(mat)x.array() + x; */
    /* cout<<x.array().inverse(); */
    net.add(make_shared<linear>(14 * 14, 128));
    net.add(make_shared<hardswish>());
    net.add(make_shared<batchnorm>(128));
    net.add(make_shared<linear>(128, 64));

    net.add(make_shared<hardswish>());
    net.add(make_shared<batchnorm>(64));
    net.add(make_shared<linear>(64, 64));

    net.add(make_shared<hardswish>());
    net.add(make_shared<linear>(64, 10));

    net.add(make_shared<softmax>());
    cout << net.shape();
    /* net.read("./newmnist.txt"); */

    vector<double> labels, _labels;
    vector<vector<double>> images, _images;
    read_Mnist_Label("./train-labels.idx1-ubyte", labels);
    read_Mnist_Label("./t10k-labels.idx1-ubyte", _labels);
    read_Mnist_Images("./train-images.idx3-ubyte", images);
    read_Mnist_Images("./t10k-images.idx3-ubyte", _images);

    /* double cnt = 0; */
    for (int i = 0; i < _images.size(); i++) {
        vector<double> out = vector<double>(10, 0);
        out[_labels[i]] = 1;
        data.valid.first.push_back(make_vec(pool(_images[i])));
        data.valid.second.push_back(make_vec(out));
        /* cnt -= chk(net.forward(make_vec(pool(_images[i]))),make_vec(out)); */
    };
    /* cout << cnt/_images.size() << endl; */
    for (int i = 0; i < images.size(); i++) {
        vector<double> out = vector<double>(10, 0);
        out[labels[i]] = 1;
        data.train.first.push_back(make_vec(pool(images[i])));
        data.train.second.push_back(make_vec(out));
    }
    /* sgd( */
    /*     data, net, 128, 50000, [](int x) { return 1. / 128 / 10; }, chk_k); */
    adam(data, net, 128, 50000, chk_k);

    /* net.sgd( */
    /*     train_data, test_data, 8, 50000, 5000, [](int x) -> double { return 1. / 8 / (1. + x * 0.005); }, 3); */
    /* net.write("newmnist.txt"); */
    return 0;
}
