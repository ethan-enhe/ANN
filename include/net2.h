#ifndef NET_H
#define NET_H

#include <bits/stdc++.h>
#include "../Eigen/Core"

using namespace Eigen;
using namespace std;

using vec = VectorXd;
using mat = MatrixXd;
using vec_batch = vector<vec>;
using batch = pair<vec_batch, vec_batch>;
const double INF = 1e8;
const double EPS = 1e-8;
const double adam_lr = 0.001, adam_rho1 = 0.9, adam_rho2 = 0.999;

//{{{ Random
std::mt19937_64 mr(std::chrono::system_clock::now().time_since_epoch().count());
double rd(double l, double r) { return std::uniform_real_distribution<double>(l, r)(mr); }
double nd(double l, double r) { return std::normal_distribution<double>(l, r)(mr); }
int ri(int l, int r) { return std::uniform_int_distribution<int>(l, r)(mr); }
//}}}
//{{{ Utils
vec make_vec(const vector<double> &x) {
    VectorXd res;
    res.resize(x.size());
    for (int i = 0; i < (int)x.size(); i++) res(i) = x[i];
    return res;
}
//}}}
//{{{ Error func
double variance(const vec_batch &out, const vec_batch &label) {
    auto square = [](double x) { return x * x; };
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res += square(out[i](j) - label[i](j));

    return res / batch_sz / 2;
}
//分2类，交叉熵
double crossentropy_2(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++)
            res -= label[i](j) * log(out[i](j)) + (1. - label[i](j)) * log(1. - out[i](j));
    return res;
}
double crossentropy_k(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res -= label[i](j) * log(out[i](j));
    return res;
}
double chk_k(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++) {
        int mx = 0;
        for (int j = 0; j < out[i].rows(); j++)
            if (out[i](j) > out[i](mx)) mx = j;
        res += (label[i](mx) > 0.9);
    }
    return res / batch_sz;
}
//}}}
//{{{ Layer
struct layer;
using layerp = shared_ptr<layer>;

//一层神经元的基类
struct layer {
    string name;
    vec_batch out, grad;
    int batch_sz;
    bool train_mode;

    layer(const string &name) : name(name), batch_sz(0), train_mode(1) {}

    //更改batch-size 和 trainmode 选项
    void set_train_mode(const bool &new_train_mode) { train_mode = new_train_mode; }
    virtual void _resize(int){};
    void resize(int new_batch_sz) {
        if (batch_sz != new_batch_sz) {
            batch_sz = new_batch_sz;
            out.resize(batch_sz);
            grad.resize(batch_sz);
            _resize(batch_sz);
        }
    }
    //把输入向量运算后放到Z里
    virtual void forward(const vec_batch &in) {}
    //根据输入向量，和答案对下一层神经元输入的偏导，算出答案对这一层神经元输入的偏导
    virtual void backward(const vec_batch &, const vec_batch &) = 0;
    //以 rate 学习率梯度下降
    virtual void sgd(double){};
    virtual void adam(int){};
};

struct linear : public layer {
    const int in_sz, out_sz;
    // 系数，系数梯度，一阶累积量，二阶累积量
    mat weight, grad_weight, s_weight, r_weight;
    vec bias, grad_bias, s_bias, r_bias;
    linear(int in_sz, int out_sz)
        : layer("linear " + to_string(in_sz) + " -> " + to_string(out_sz))
        , in_sz(in_sz)
        , out_sz(out_sz)
        , weight(out_sz, in_sz)
        , grad_weight(out_sz, in_sz)
        , s_weight(out_sz, in_sz)
        , r_weight(out_sz, in_sz)
        , bias(out_sz)
        , grad_bias(out_sz)
        , s_bias(out_sz)
        , r_bias(out_sz) {
        s_weight.setZero();
        r_weight.setZero();
        s_bias.setZero();
        r_bias.setZero();
        for (auto &i : weight.reshaped()) i = nd(0, 2. / in_sz);
    }
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = weight * in[i] + bias;
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        // nx.in==W*this.in+B
        // dnx.in/din
        grad_weight.setZero();
        grad_bias.setZero();
        for (int i = 0; i < batch_sz; i++) {
            grad_bias += nxt_grad[i];
            grad_weight += nxt_grad[i] * in[i].transpose();
            grad[i] = weight.transpose() * nxt_grad[i];
        }
    }
    void adam(int t) {
        double inv1 = 1. / (1. - pow(adam_rho1, t));
        double inv2 = 1. / (1. - pow(adam_rho2, t));
        s_weight = adam_rho1 * s_weight + (1. - adam_rho1) * grad_weight;
        r_weight = adam_rho2 * r_weight + (1. - adam_rho2) * grad_weight.cwiseAbs2();
        weight.array() -= inv1 * adam_lr * s_weight.array() / (sqrt(inv2 * r_weight.array()) + EPS);

        s_bias = adam_rho1 * s_bias + (1. - adam_rho1) * grad_bias;
        r_bias = adam_rho2 * r_bias + (1. - adam_rho2) * grad_bias.cwiseAbs2();
        bias.array() -= inv1 * adam_lr * s_bias.array() / (sqrt(inv2 * r_bias.array()) + EPS);
    }
    void sgd(double rate) {
        bias -= rate * grad_bias;
        weight -= rate * grad_weight;
    }
};
struct sigmoid : public layer {
    sigmoid() : layer("sigmoid") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](double x) { return 1. / (exp(-x) + 1); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(out[i].unaryExpr([](double x) { return x * (1. - x); }));
    }
};
struct th : public layer {
    th() : layer("tanh") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](double x) { return tanh(x); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(out[i].unaryExpr([](double x) { return 1. - x * x; }));
    }
};
struct relu : public layer {
    relu() : layer("relu") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](double x) { return max(0., x); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](double x) -> double { return x < 0 ? 0 : 1; }));
    }
};
struct hardswish : public layer {
    hardswish() : layer("hardswish") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++)
            out[i] = in[i].unaryExpr([](double x) -> double {
                if (x >= 3) return x;
                if (x <= -3) return 0;
                return x * (x + 3) / 6;
            });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](double x) -> double {
                if (x >= 3) return 1;
                if (x <= -3) return 0;
                return x / 3 + 0.5;
            }));
    }
};
struct softmax : public layer {
    softmax() : layer("softmax") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) {
            out[i].resize(in[i].size());
            double sum = 0;
            for (int j = 0; j < (int)in[i].size(); j++) sum += (out[i](j) = exp(in[i](j)));
            out[i] /= sum;
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) { assert(0); }
};
struct batchnorm : public layer {
    //平均值，方差
    vec mean, running_mean, grad_mean;
    vec var, running_var, grad_var, inv_var;
    vec gama, grad_gama, s_gama, r_gama;
    vec beta, grad_beta, s_beta, r_beta;
    //这两个用来辅助,inv记录1/sqrt(方差+eps)
    vec grad_normalized_x;
    const double momentum;
    batchnorm(int sz, double momentum = 0.9)
        : layer("batchnorm_sgd " + to_string(sz))
        , mean(sz)
        , running_mean(sz)
        , grad_mean(sz)
        , var(sz)
        , running_var(sz)
        , grad_var(sz)
        , inv_var(sz)
        , gama(sz)
        , grad_gama(sz)
        , s_gama(sz)
        , r_gama(sz)
        , beta(sz)
        , grad_beta(sz)
        , s_beta(sz)
        , r_beta(sz)
        , momentum(momentum) {
        gama.setOnes();
        beta.setZero();
        s_gama.setZero();
        r_gama.setZero();
        s_beta.setZero();
        r_beta.setZero();
    }
    void forward(const vec_batch &in) {
        if (train_mode) {
            mean.setZero();
            var.setZero();
            for (int i = 0; i < batch_sz; i++) mean += in[i];
            mean /= batch_sz;
            for (int i = 0; i < batch_sz; i++) var += (in[i] - mean).cwiseAbs2();
            var /= batch_sz;
            inv_var = rsqrt(var.array() + EPS);
            running_mean = running_mean * momentum + mean * (1 - momentum);
            running_var = running_var * momentum + var * (1 - momentum);

            for (int i = 0; i < batch_sz; i++)
                out[i] = (in[i] - mean).array() * inv_var.array() * gama.array() + beta.array();
        } else {
            for (int i = 0; i < batch_sz; i++)
                out[i] =
                    (in[i] - running_mean).array() * rsqrt(running_var.array() + EPS) * gama.array() + beta.array();
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        grad_gama.setZero();
        grad_beta.setZero();
        grad_mean.setZero();
        grad_var.setZero();
        for (int i = 0; i < batch_sz; i++) {
            grad_normalized_x = nxt_grad[i].array() * gama.array();

            grad_var.array() += grad_normalized_x.array() * (in[i] - mean).array();
            grad_mean.array() += grad_normalized_x.array();

            grad[i] = grad_normalized_x.array() * inv_var.array();

            grad_beta.array() += nxt_grad[i].array();
            grad_gama.array() += nxt_grad[i].array() * (in[i] - mean).array() * inv_var.array();
        }
        grad_var = -0.5 * grad_var.array() * pow(inv_var.array(), 3);
        grad_mean = -grad_mean.array() * inv_var.array();
        for (int i = 0; i < batch_sz; i++)
            grad[i].array() += (grad_mean.array() + 2 * grad_var.array() * (in[i] - mean).array()) / batch_sz;
    }
    void sgd(double rate) {
        grad_beta -= rate * grad_beta;
        grad_gama -= rate * grad_gama;
    }
    void adam(int t) {
        double inv1 = 1. / (1. - pow(adam_rho1, t));
        double inv2 = 1. / (1. - pow(adam_rho2, t));
        s_beta = adam_rho1 * s_beta + (1. - adam_rho1) * grad_beta;
        r_beta = adam_rho2 * r_beta + (1. - adam_rho2) * grad_beta.cwiseAbs2();
        beta.array() -= inv1 * adam_lr * s_beta.array() / (sqrt(inv2 * r_beta.array()) + EPS);

        s_gama = adam_rho1 * s_gama + (1. - adam_rho1) * grad_gama;
        r_gama = adam_rho2 * r_gama + (1. - adam_rho2) * grad_gama.cwiseAbs2();
        gama.array() -= inv1 * adam_lr * s_gama.array() / (sqrt(inv2 * r_gama.array()) + EPS);
    }
};
struct same : public layer {
    same() : layer("same") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i];
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) grad[i] = nxt_grad[i];
    }
};
//}}}
//{{{ Layers Sequnce
struct layer_seq {
    int batch_sz;
    vector<layerp> layers;
    layer_seq() : batch_sz(0) {
        layerp x = make_shared<same>();
        x->name = "input";
        layers.emplace_back(x);
    }
    void add(const layerp &x) { layers.push_back(x); }
    string shape() {
        string res = "";
        for (auto &it : layers) res += it->name + "\n";
        return res;
    }
    void set_train_mode(const bool &new_train_mod) {
        for (auto &l : layers) l->set_train_mode(new_train_mod);
    }

    vec_batch forward(const vec_batch &input) {
        if ((int)input.size() != batch_sz) {
            batch_sz = input.size();
            for (auto &l : layers) l->resize(batch_sz);
        }
        int layer_sz = layers.size();
        layers[0]->forward(input);
        for (int i = 1; i < layer_sz; i++) layers[i]->forward(layers[i - 1]->out);
        return layers.back()->out;
    }
    vec_batch backward(const vec_batch &label) {
        for (int i = 0; i < batch_sz; i++) layers.back()->grad[i] = layers.back()->out[i] - label[i];
        int layer_sz = layers.size();
        for (int i = layer_sz - 2; i >= 0; i--)
            layers[i]->backward(i ? layers[i - 1]->out : vec_batch(), layers[i + 1]->grad);
        return layers[0]->grad;
    }
    void sgd(const batch &data, double rate) {
        forward(data.first);
        backward(data.second);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->sgd(rate);
    }
    void adam(const batch &data, int t) {
        forward(data.first);
        backward(data.second);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->adam(t);
    }
};
//}}}
//{{{ Dataset
struct data_set {
    batch train, valid, test;
    data_set() {}
    data_set(const batch &all_data) {
        for (int i = 0; i < (int)all_data.first.size(); i++) {
            int rnd = ri(0, 10);
            if (rnd == 0) {
                test.first.push_back(all_data.first[i]);
                test.second.push_back(all_data.second[i]);
            } else if (rnd == 1) {
                valid.first.push_back(all_data.first[i]);
                valid.second.push_back(all_data.second[i]);
            } else {
                train.first.push_back(all_data.first[i]);
                train.second.push_back(all_data.second[i]);
            }
        }
    }
    batch get_train_batch(int batch_sz) const {
        assert(train.first.size());
        batch res;
        for (int i = 0; i < batch_sz; i++) {
            int id = ri(0, train.first.size() - 1);
            res.first.push_back(train.first[id]);
            res.second.push_back(train.second[id]);
        }
        return res;
    }
    batch get_valid_batch(int batch_sz) const {
        assert(valid.first.size());
        batch res;
        for (int i = 0; i < batch_sz; i++) {
            int id = ri(0, valid.first.size() - 1);
            res.first.push_back(valid.first[id]);
            res.second.push_back(valid.second[id]);
        }
        return res;
    }
};
//}}}
//{{{ Trainning

void sgd(const data_set &data, layer_seq &net, int batch_sz, int epoch, function<double(int)> lr_func,
         function<double(const vec_batch &, const vec_batch &)> err_func) {
    int t0 = clock();
    for (int i = 1; i <= epoch; i++) {
        net.sgd(data.get_train_batch(batch_sz), lr_func(i));
        if (i % 50 == 0) {
            net.set_train_mode(0);
            double sum = 0;
            for (int j = 0; j < (int)data.valid.first.size(); j++) {
                batch tmp = {{data.valid.first[j]}, {data.valid.second[j]}};
                sum += err_func(net.forward(tmp.first), tmp.second);
            }
            cerr << "Time elapse: " << (double)(clock() - t0) / CLOCKS_PER_SEC << endl;
            cerr << "Epoch: " << i << endl;
            cerr << "Error: " << sum / data.valid.first.size() << endl;
            net.set_train_mode(1);
        }
    }
}
void adam(const data_set &data, layer_seq &net, int batch_sz, int epoch,
          function<double(const vec_batch &, const vec_batch &)> err_func) {
    int t0 = clock();
    for (int i = 1; i <= epoch; i++) {
        net.adam(data.get_train_batch(batch_sz), i);
        if (i % 50 == 0) {
            net.set_train_mode(0);
            double sum = 0;
            for (int j = 0; j < (int)data.valid.first.size(); j++) {
                batch tmp = {{data.valid.first[j]}, {data.valid.second[j]}};
                sum += err_func(net.forward(tmp.first), tmp.second);
            }
            cerr << "Time elapse: " << (double)(clock() - t0) / CLOCKS_PER_SEC << endl;
            cerr << "Epoch: " << i << endl;
            cerr << "Error: " << sum / data.valid.first.size() << endl;
            net.set_train_mode(1);
        }
    }
}

//}}}

#endif
