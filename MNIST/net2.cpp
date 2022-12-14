#include <bits/stdc++.h>

#include "../include/microdnn.h"

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
        cerr << "magic number = " << magic_number << endl;
        cerr << "number of images = " << number_of_images << endl;

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
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        cerr << "magic number = " << magic_number << endl;
        cerr << "number of images = " << number_of_images << endl;
        cerr << "rows = " << n_rows << endl;
        cerr << "cols = " << n_cols << endl;

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

sequential net;
int main() {
    net.add(make_shared<linear>(28 * 28, 300));
    net.add(make_shared<batchnorm>(300));

    net.add(make_shared<relu>());
    net.add(make_shared<linear>(300, 128));

    net.add(make_shared<relu>());
    net.add(make_shared<batchnorm>(128));
    net.add(make_shared<linear>(128, 64));

    net.add(make_shared<relu>());
    net.add(make_shared<linear>(64, 10));

    net.add(make_shared<softmax>());


    net.readf("model_acc98.71.txt");
    net.set_train_mode(0);
    cerr << net.shape();

    vector<double> labels, _labels;
    vector<vector<double>> images, _images;
    read_Mnist_Label("./train-labels.idx1-ubyte", labels);
    read_Mnist_Label("./t10k-labels.idx1-ubyte", _labels);
    read_Mnist_Images("./train-images.idx3-ubyte", images);
    read_Mnist_Images("./t10k-images.idx3-ubyte", _images);

data_set dat;
    double cnt = 0;
    net.set_train_mode(0);
    for (int i = 0; i < (int)_images.size(); i++) {
        vector<double> out = vector<double>(10, 0);
        out[_labels[i]] = 1;
        dat.valid.first.push_back(make_vec(_images[i]));
        dat.valid.second.push_back(make_vec(out));
        // cnt += chk_k(net.forward({make_vec(_images[i])}), {make_vec(out)});
    };
    // cout << cnt / _images.size() << endl;
    // freopen("data.out", "w", stdout);
    // while (1) {
    //     int id = 0;
    //     cin >> id;
    //     id = ri(0, dat.valid.first.size() - 1);
    //     for (int i = 0; i < 28; i++, putchar('\n'))
    //         for (int j = 0; j < 28; j++) printf("%4d ", (int)dat.valid.first[id](i * 28 + j));
    //     putchar('\n');
    // }
    net.set_train_mode(1);

    for (int i = 0; i < (int)images.size(); i++) {
        vector<double> out = vector<double>(10, 0);
        out[labels[i]] = 1;
        dat.train.first.push_back(make_vec(images[i]));
        dat.train.second.push_back(make_vec(out));
    }

    nesterov opt(0.01, 0.9);
    upd(opt, dat, net, 64, 100000, chk_k, "model.txt");

    return 0;
}
