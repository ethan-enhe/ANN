#ifndef NET_H
#define NET_H

#include <bits/stdc++.h>

#include <algorithm>
#include <vector>

#include "../include/tools.h"

struct layer;
using layerp = shared_ptr<layer>;

bool train_mode = 0;

//一层神经元的基类
struct layer {
    string name;
    vec_batch out, grad;
    int batch_sz;
    bool train_mode;

    layer(const string &name) : name(name), batch_sz(0), train_mode(1) {}

    //更改batch-size 和 trainmode 选项
    virtual void _attri(const int &, const bool &){};
    void attri(const int &new_batch_sz, const bool &new_train_mod) {
        batch_sz = new_batch_sz;
        train_mode = new_train_mod;
        out.resize(batch_sz);
        grad.resize(batch_sz);
        _attri(batch_sz, train_mode);
    }
    //把输入向量运算后放到Z里
    virtual void forward(const vec_batch &in) {}
    //根据输入向量，和答案对下一层神经元输入的偏导，算出答案对这一层神经元输入的偏导
    virtual void backward(const vec_batch &, const vec_batch &) = 0;
    //以 rate 学习率更新系数。
    virtual void update(double){};
};

struct linear : public layer {
    const int in_sz, out_sz;
    mat weight, grad_weight;
    vec bias, grad_bias;
    linear(const int in_sz, const int out_sz)
        : layer("linear " + to_string(in_sz) + " * " + to_string(out_sz))
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
    void update(double rate) {
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
struct same : public layer {
    same() : layer("same") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i];
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) grad[i] = nxt_grad[i];
    }
};

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

    vec_batch forward(const vec_batch &input) {
        if ((int)input.size() != batch_sz) {
            batch_sz = input.size();
            for (auto &l : layers) l->attri(batch_sz, l->train_mode);
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
        for (int i = 0; i < layer_sz; i++) layers[i]->update(rate);
    }
};

using batch = pair<vec_batch, vec_batch>;

struct data_set {
    vec_batch train, valid, test;
    data_set(const batch &all_data,int valid_num,int test_num) {

    }
};

#endif
