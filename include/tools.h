#ifndef TOOLS_H
#define TOOLS_H
#include <bits/stdc++.h>

#include <vector>

#include "../Eigen/Dense"
using namespace Eigen;
using namespace std;
using vec = VectorXd;
using mat = MatrixXd;
using vec_batch = vector<vec>;

std::mt19937_64 mr(std::chrono::system_clock::now().time_since_epoch().count());
// std::mt19937_64 mr(1923801);
double rd(double l, double r) { return std::uniform_real_distribution<double>(l, r)(mr); }
double nd(double l, double r) { return std::normal_distribution<double>(l, r)(mr); }
int ri(int l, int r) { return std::uniform_int_distribution<int>(l, r)(mr); }

vec make_vec(const vector<double> &x) {
    VectorXd res;
    res.resize(x.size());
    for (int i = 0; i < (int)x.size(); i++) res(i) = x[i];
    return res;
}

double varience(const vec_batch &out, const vec_batch &label) {
    auto square = [](double x) { return x * x; };
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res += square(out[i](j) - label[i](j));

    return res / 2 / batch_sz;
}
//分2类，交叉熵
double crossentropy_2(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++)
            res -= label[i](j) * std::log(out[i](j)) + (1. - label[i](j)) * std::log(1. - out[i](j));
    return res;
}
#endif
