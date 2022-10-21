#ifndef NET_H
#define NET_H

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")
#include <bits/stdc++.h>

#include "../Eigen/Dense"
#include "../include/rnd.h"
using namespace Eigen;

double hardswish(double x) {
    if (x >= 3) return x;
    if (x <= -3) return 0;
    return x * (x + 3) / 6;
}
double hardswish_da(double x) {
    if (x >= 3) return 1;
    if (x <= -3) return 0;
    return x / 3 + 0.5;
}
double swish(double x) { return x / (std::exp(-x) + 1); }
double swish_da(double x) {
    double ex = std::exp(x);
    return ex * (x + ex + 1) / (ex + 1) / (ex + 1);
}
double mish(double x) { return x * std::tanh(std::log(1 + std::exp(x))); }
double mish_da(double x) {
    double ex = std::exp(x);
    double fm = (2 * ex + ex * ex + 2);
    return ex * (4 * (x + 1) + 4 * ex * ex + ex * ex * ex + ex * (4 * x + 6)) / fm / fm;
}
double sigmoid(double x) { return 1. / (std::exp(-x) + 1); }
double sigmoid_dz(double y) { return y * (1. - y); }
double relu(double x) { return std::max(0., x); }
double relu_dz(double y) { return y <= 1e-6 ? 0 : 1; }
double relu_da(double x) { return x <= 0 ? 0 : 1; }
double square(double x) { return x * x; }

struct BP {
    // output weight bias gradient
    std::vector<VectorXd> A, Z, W0, G;
    std::vector<MatrixXd> W;
    void init(const std::vector<int> &width) {
        A.resize(width.size());
        Z.resize(width.size());
        W.resize(width.size());
        W0.resize(width.size());
        G.resize(width.size());
        for (int i = 0; i < (int)width.size(); i++) {
            A[i] = VectorXd::Zero(width[i]);
            Z[i] = VectorXd::Zero(width[i]);
            G[i] = VectorXd::Zero(width[i]);
            if (i) {
                /* Xavier初始化，函数为sigmoid时使用 */
                /* W[i] = MatrixXd::Random(width[i], width[i - 1]) * sqrt(6. / (width[i - 1] + width[i])); */
                /* Relu函数对应的初始化 */
                W[i].resize(width[i], width[i - 1]);
                for (int x = 0; x < width[i]; x++)
                    for (int y = 0; y < width[i - 1]; y++) W[i](x, y) = nd(0, 2. / width[i - 1]);
                W0[i] = VectorXd::Zero(width[i]);
            }
        }
    }
    friend std::ostream &operator<<(std::ostream &io, const BP &x) {
        io << x.Z.size() << '\n';
        for (int i = 0; i < (int)x.Z.size(); i++) io << x.Z[i].size() << ' ';
        io << '\n';
        for (int i = 1; i < (int)x.Z.size(); i++) io << x.W[i] << '\n' << x.W0[i] << '\n';
        return io;
    }
    friend std::istream &operator>>(std::istream &io, BP &x) {
        int dpth;
        io >> dpth;
        std::vector<int> width(dpth);
        for (int &i : width) io >> i;
        x.init(width);
        for (int i = 1; i < dpth; i++) {
            for (int j = 0; j < width[i]; j++)
                for (int k = 0; k < width[i - 1]; k++) io >> x.W[i](j, k);
            for (int j = 0; j < width[i]; j++) io >> x.W0[i](j);
        }
        return io;
    }
    std::vector<double> run(const std::vector<double> &input) {
        for (int i = 0; i < (int)input.size(); i++) Z[0](i) = input[i];
        for (int i = 1; i < (int)W.size(); i++) {
            A[i] = (W[i] * Z[i - 1] + W0[i]);
            //输出层还是用sigmoid
            if (i + 1 < (int)W.size())
                Z[i] = A[i].unaryExpr(std::ref(relu));
            else
                Z[i] = A[i].unaryExpr(std::ref(sigmoid));
        }
        std::vector<double> ans;
        for (int i = 0; i < Z.back().size(); i++) ans.push_back(Z.back()(i));
        return ans;
    }
    void delt(const std::vector<double> &output) {
        for (int i = 0; i < (int)output.size(); i++) G.back()(i) = output[i] - Z.back()(i);
        /* 误差的定义与方差有所不同，如果定义为方差，则应该这么搞 */
        /* G.back()=G.back().mult(Z.back().f(_sigmoid)); */
        for (int i = (int)W.size() - 2; i >= 1; i--)
            G[i] = A[i].unaryExpr(std::ref(relu_da)).cwiseProduct((W[i + 1].transpose()) * G[i + 1]);
        G[0] = W[1].transpose() * G[1];
    }
    void tra(double rate) {
        for (int i = (int)W.size() - 1; i; i--) {
            W[i] += G[i] * (Z[i - 1].transpose()) * rate;
            W0[i] += G[i] * rate;
        }
    }
    double err(const std::vector<double> &output) {
        double res = 0;
        for (int i = 0; i < (int)output.size(); i++) res += square(Z.back()(i) - output[i]);
        return sqrt(res / output.size());
    }
    bool chk(const std::vector<double> &output) {
        int mx = 0;
        for (int i = 0; i < (int)output.size(); i++)
            if (Z.back()(i) > Z.back()(mx)) mx = i;
        return output[mx] >= 0.8;
    }

    void train_all(std::vector<std::pair<std::vector<double>, std::vector<double>>> &train_data,
                   std::vector<std::pair<std::vector<double>, std::vector<double>>> &test_data, double rate = 0.005,
                   int totalt = 1e6, int prtt = 1e3) {
        for (int i = 1; i <= totalt; i++) {
            int id = ri(0, train_data.size() - 1);
            run(train_data[id].first);
            delt(train_data[id].second);
            tra(1. / (1. + i * rate));

            if (i % prtt == 0) {
                double sum = 0;
                for (int j = 0; j <= prtt; j++) {
                    int id = ri(0, test_data.size() - 1);
                    run(test_data[id].first);
                    /* sum += chk(test_data[id].second); */
                    sum += err(test_data[id].second);
                }
                std::cerr << "Training progress: " << (double)i / totalt * 100 << "% Accuracy: " << sum / prtt * 100
                          << "%" << std::endl;
            }
        }
    }
};
#endif
