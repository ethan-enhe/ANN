#ifndef NET_H
#define NET_H
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")
#include <bits/stdc++.h>

#include "../Eigen/Dense"
#include "../include/rnd.h"

using namespace Eigen;
using namespace std;
struct layer;
using layerp = shared_ptr<layer>;
using dataset = vector<pair<VectorXd, VectorXd>>;

//小工具：std::vector->Eigen::VectorXd
VectorXd std2eigen(const vector<double>& x) {
    VectorXd res;
    res.resize(x.size());
    for (int i = 0; i < (int)x.size(); i++) res(i) = x[i];
    return res;
}

//权系数矩阵，包括偏移，封装了一下，便于实现adam
struct mat_weight {
    MatrixXd W;
    VectorXd B;
    void init(int w, int _w) {
        W = MatrixXd::Zero(w, _w);
        B = VectorXd::Zero(w);
    }
    void clear() {
        W.setZero();
        B.setZero();
    }

    VectorXd forward(const VectorXd& input) const { return W * input + B; }
    VectorXd backward(const VectorXd& input) const { return W.transpose() * input; }
    mat_weight& operator+=(const mat_weight& y) {
        W += y.W;
        B += y.B;
        return *this;
    }
    mat_weight& operator-=(const mat_weight& y) {
        W -= y.W;
        B -= y.B;
        return *this;
    }
    mat_weight operator*(const double& y) const { return {W * y, B * y}; }
    mat_weight cwiseProduct(const mat_weight& y) const { return {W.cwiseProduct(y.W), B.cwiseProduct(y.B)}; }
    mat_weight unaryExpr(const function<double(double)>& f) const { return {W.unaryExpr(f), B.unaryExpr(f)}; }
    friend std::ostream& operator<<(std::ostream& io, const mat_weight& x) {
        io << x.B << '\n';
        io << x.W << '\n';
        return io;
    }
    friend std::istream& operator>>(std::istream& io, mat_weight& x) {
        for (auto& i : x.B) io >> i;
        for (int i = 0; i < x.W.rows(); i++)
            for (int j = 0; j < x.W.cols(); j++) io >> x.W(i, j);
        return io;
    }
};

//一层神经元的基类
struct layer {
    VectorXd A, Z, GA;
    mat_weight W, GW;
    layer(const layerp& pre, int w) {
        A.resize(w);
        Z.resize(w);
        GA.resize(w);
        if (pre != nullptr) {
            W.init(w, pre->Z.size());
            GW.init(w, pre->Z.size());
            for (auto& i : W.W.reshaped()) i = nd(0, 2. / pre->Z.size());
        }
    }

    //将A激活，复制到Z上
    virtual void acti() = 0;
    //激活函数对A_i求导，复制到G上
    virtual void deri() = 0;
    void forward(const layerp& pre) {
        A = W.forward(pre->Z);
        acti();
    }
    void backward(const layerp& pre, const layerp& nxt) {
        if (nxt != nullptr) {
            deri();
            GA = GA.cwiseProduct(nxt->W.backward(nxt->GA));
        }
        if (pre != nullptr) GW += {GA * pre->Z.transpose(), GA};
    }
    VectorXd output() const { return Z; }
    VectorXd gradient() const { return GA; }
};

struct relu : public layer {
    using layer::layer;
    void acti() {
        Z = A.unaryExpr([](double x) -> double { return std::max(0., x); });
    }
    void deri() {
        GA = A.unaryExpr([](double x) -> double { return x <= 0 ? 0 : 1; });
    }
};
struct hardswish : public layer {
    using layer::layer;
    void acti() {
        Z = A.unaryExpr([](double x) -> double {
            if (x >= 3) return x;
            if (x <= -3) return 0;
            return x * (x + 3) / 6;
        });
    }
    void deri() {
        GA = A.unaryExpr([](double x) -> double {
            if (x >= 3) return 1;
            if (x <= -3) return 0;
            return x / 3 + 0.5;
        });
    }
};
struct swish : public layer {
    using layer::layer;
    void acti() {
        Z = A.unaryExpr([](double x) -> double { return x / (std::exp(-x) + 1); });
    }
    void deri() {
        GA = A.unaryExpr([](double x) -> double {
            double ex = std::exp(x);
            return ex * (x + ex + 1) / (ex + 1) / (ex + 1);
        });
    }
};
struct mish : public layer {
    using layer::layer;
    void acti() {
        Z = A.unaryExpr([](double x) -> double { return x * std::tanh(std::log(1 + std::exp(x))); });
    }
    void deri() {
        GA = A.unaryExpr([](double x) -> double {
            double ex = std::exp(x);
            double fm = (2 * ex + ex * ex + 2);
            return ex * (4 * (x + 1) + 4 * ex * ex + ex * ex * ex + ex * (4 * x + 6)) / fm / fm;
        });
    }
};
struct sigmoid : public layer {
    using layer::layer;
    void acti() {
        Z = A.unaryExpr([](double x) { return 1. / (std::exp(-x) + 1); });
    }
    void deri() {
        GA = Z.unaryExpr([](double x) { return x * (1. - x); });
    }
};
struct same : public layer {
    using layer::layer;
    void acti() { Z = A; }
    void deri() { GA.setOnes(); }
};
struct softmax : public layer {
    using layer::layer;
    void acti() {
        double sum = 0;
        for (int i = 0; i < Z.rows(); i++) sum += (Z(i) = std::exp(A(i)));
        Z /= sum;
    }
    void deri() { assert(0); }
};

struct ANN {
    vector<layerp> layers;
    friend std::ostream& operator<<(std::ostream& io, const ANN& x) {
        for (const auto& i : x.layers) io << i->W;
        return io;
    }
    friend std::istream& operator>>(std::istream& io, ANN& x) {
        for (const auto& i : x.layers) io >> i->W;
        return io;
    }
    //加一层
    template <typename T, typename T1, typename... T2>
    void add(T1 x, T2... y) {
        if (layers.empty())
            layers.emplace_back(make_shared<T>(nullptr, x, y...));
        else
            layers.emplace_back(make_shared<T>(layers.back(), x, y...));
    }
    VectorXd forward(const VectorXd& input) {
        layers.front()->A = input;
        layers.front()->acti();
        for (int i = 1; i < (int)layers.size(); i++) layers[i]->forward(layers[i - 1]);
        return layers.back()->Z;
    }
    VectorXd backward(const VectorXd& output) {
        layers.back()->GA = layers.back()->Z - output;
        for (int i = (int)layers.size() - 1; i >= 0; i--) {
            layers[i]->backward(i ? layers[i - 1] : nullptr, i + 1 == (int)layers.size() ? nullptr : layers[i + 1]);
        }
        return layers.front()->GA;
    }
    //方差
    double variance(const VectorXd& output) {
        auto square = [](double x) { return x * x; };
        double ans = 0;
        for (int i = 0; i < output.rows(); i++) ans += square(layers.back()->Z(i) - output(i));
        return std::sqrt(ans / output.rows());
    }
    //分k类，交叉熵
    double crossentropy_k(const VectorXd& output) {
        double ans = 0;
        for (int i = 0; i < output.rows(); i++) ans -= output(i) * std::log(layers.back()->Z(i));
        return ans;
    }
    //分2类，交叉熵
    double crossentropy(const VectorXd& output) {
        double ans = 0;
        for (int i = 0; i < output.rows(); i++)
            ans -= output(i) * std::log(layers.back()->Z(i)) + (1. - output(i)) * std::log(1. - layers.back()->Z(i));
        return ans;
    }
    //分k类，正确率
    double chk(const VectorXd& output) {
        int mx = 0;
        for (int i = 0; i < output.rows(); i++)
            if (layers.back()->Z(i) > layers.back()->Z(mx)) mx = i;
        return output[mx] > 0.6;
    }
    //小批量sgd
    void sgd(const dataset& train, const dataset& test, int batch, int tottime, int prttime,
             const function<double(double)>& rate, int errtype = 0) {
        assert(!train.empty());
        for (int i = 1; i <= tottime; i++) {
            for (int i = 1; i < (int)layers.size(); i++) layers[i]->GW.clear();
            for (int j = 1; j <= batch; j++) {
                int id = ri(0, (int)train.size() - 1);
                forward(train[id].first);
                backward(train[id].second);
            }
            double r = rate(i);
            for (int i = 1; i < (int)layers.size(); i++) layers[i]->W -= layers[i]->GW * r;

            if (i % prttime == 0) {
                double sum = 0;
                for (int i = 0; i < (int)test.size(); i++) {
                    forward(test[i].first);
                    switch (errtype) {
                        case 0: sum += variance(test[i].second);
                        case 1: sum += crossentropy(test[i].second);
                        case 2: sum += crossentropy_k(test[i].second);
                        case 3: sum += chk(test[i].second);
                    }
                }
                cerr << "Training progress: " << (double)i / tottime * 100 << "% Accuracy: " << sum / test.size()
                     << endl;
            }
        }
    }
};
#endif
