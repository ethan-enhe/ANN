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
    void set_train_mod(const bool &new_train_mod) { train_mode = new_train_mod; }
    virtual void _resize(const int &){};
    void resize(const int &new_batch_sz) {
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
    virtual void update_with_grad(double){};
};

struct linear : public layer {
    const int in_sz, out_sz;
    mat weight, grad_weight;
    vec bias, grad_bias;
    linear(const int &in_sz, const int &out_sz)
        : layer("linear " + to_string(in_sz) + " -> " + to_string(out_sz))
        , in_sz(in_sz)
        , out_sz(out_sz)
        , weight(out_sz, in_sz)
        , grad_weight(out_sz, in_sz)
        , bias(out_sz)
        , grad_bias(out_sz) {
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
    void update_with_grad(double rate) {
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
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](double x) { return 1. / (exp(-x) + 1); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(out[i].unaryExpr([](double x) { return x * (1. - x); }));
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
    //平均值，1/sqrt(方差+eps)
    vec avg, inv_var;
    vec gama, beta;
    //这两个用来辅助
    vec grad_normalized_x, grad_avg, grad_var;
    //这两个是要拿来更新系数的
    vec grad_gama, grad_beta;
    batchnorm(const int &sz)
        : layer("batchnorm " + to_string(sz))
        , avg(sz)
        , inv_var(sz)
        , gama(sz)
        , beta(sz)
        , grad_avg(sz)
        , grad_var(sz)
        , grad_gama(sz)
        , grad_beta(sz) {
        gama.setOnes();
        beta.setZero();
    }
    void forward(const vec_batch &in) {
        if (train_mode) {
            avg.setZero();
            inv_var.setZero();
            for (int i = 0; i < batch_sz; i++) avg += in[i];
            avg /= batch_sz;
            for (int i = 0; i < batch_sz; i++) inv_var += (in[i] - avg).cwiseAbs2();
            inv_var = ((inv_var / batch_sz).array() + EPS).cwiseSqrt().cwiseInverse();
        }
        for (int i = 0; i < batch_sz; i++) out[i] = ((in[i] - avg).cwiseProduct(inv_var)).cwiseProduct(gama) + beta;
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        grad_gama.setZero();
        grad_beta.setZero();
        grad_avg.setZero();
        grad_var.setZero();
        for (int i = 0; i < batch_sz; i++) {
            grad_normalized_x = nxt_grad[i].cwiseProduct(gama);
            grad_var += grad_normalized_x.cwiseProduct(in[i] - avg);
            grad_avg += grad_normalized_x;
            grad[i] = grad_normalized_x.cwiseProduct(inv_var);

            grad_beta += nxt_grad[i];
            grad_gama += nxt_grad[i].cwiseProduct((in[i] - avg).cwiseProduct(inv_var));
        }
        /* cout << inv_var << endl; */
        /* cout << (vec)(inv_var.array().pow(3)) << endl; */
        grad_var = -0.5 * grad_var.cwiseProduct((vec)(inv_var.array().pow(3)));
        grad_avg = -grad_avg.cwiseProduct(inv_var);
        for (int i = 0; i < batch_sz; i++) grad[i] += (grad_avg + 2 * grad_var.cwiseProduct(in[i] - avg)) / batch_sz;
        /* cout << "grad_beta:\n" << grad_avg << endl; */
        /* avg(0) += 0.001; */
    }
    void update_with_grad(double rate) {
        grad_beta -= rate * grad_beta;
        grad_gama -= rate * grad_gama;
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
    void set_train_mod(const bool &new_train_mod) {
        for (auto &l : layers) l->set_train_mod(new_train_mod);
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
    void upd(const double &rate) {
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->update_with_grad(rate);
    }
    void train(const batch &data, const double &rate) {
        forward(data.first);
        backward(data.second);
        upd(rate);
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
    batch get_train_batch(const int &batch_sz) const {
        assert(train.first.size());
        batch res;
        for (int i = 0; i < batch_sz; i++) {
            int id = ri(0, train.first.size() - 1);
            res.first.push_back(train.first[id]);
            res.second.push_back(train.second[id]);
        }
        return res;
    }
    batch get_valid_batch(const int &batch_sz) const {
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

void sgd(const data_set &data, layer_seq &net, const int &batch_sz, const int &epoch, function<double(int)> lr_func,
         function<double(const vec_batch &, const vec_batch &)> err_func) {
    int t0 = clock();
    for (int i = 1; i <= epoch; i++) {
        net.train(data.get_train_batch(batch_sz), lr_func(i));
        if (i % 5000 == 0) {
            net.set_train_mod(0);
            double sum = 0;
            for (int j = 0; j < (int)data.valid.first.size(); j++) {
                batch tmp = {{data.valid.first[j]}, {data.valid.second[j]}};
                sum += err_func(net.forward(tmp.first), tmp.second);
            }
            cerr << "Time elapse: " << (double)(clock() - t0) / CLOCKS_PER_SEC << endl;
            cerr << "Epoch: " << i << endl;
            cerr << "Error: " << sum / data.valid.first.size() << endl;
            net.set_train_mod(1);
        }
    }
}

//}}}

#endif
