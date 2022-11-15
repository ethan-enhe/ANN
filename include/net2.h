#ifndef NET_H
#define NET_H

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")

#include <bits/stdc++.h>

#include "../Eigen/Core"

using namespace Eigen;
using namespace std;

using vec = VectorXd;
using mat = MatrixXd;
using vec_batch = vector<vec>;
using batch = pair<vec_batch, vec_batch>;
const double INF = 100;
const double EPS = 1e-8;

//{{{ Random
mt19937_64 mr(chrono::system_clock::now().time_since_epoch().count());
double rd(double l, double r) { return uniform_real_distribution<double>(l, r)(mr); }
double nd(double l, double r) { return normal_distribution<double>(l, r)(mr); }
int ri(int l, int r) { return uniform_int_distribution<int>(l, r)(mr); }
//}}}
//{{{ Utils
vec make_vec(const vector<double> &x) {
    VectorXd res;
    res.resize(x.size());
    for (int i = 0; i < (int)x.size(); i++) res(i) = x[i];
    return res;
}
//}}}
//{{{ Error Function
double mylog(double x) {
    if (x == 0) return -INF;
    return log(x);
}
double variance(const vec_batch &out, const vec_batch &label) {
    auto square = [](double x) { return x * x; };
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res += square(out[i](j) - label[i](j));

    return res / batch_sz / out[0].rows();
}
double sqrtvariance(const vec_batch &out, const vec_batch &label) {
    auto square = [](double x) { return x * x; };
    const int batch_sz = out.size();
    double res = 0, ans = 0;
    for (int i = 0; i < batch_sz; i++) {
        for (int j = 0; j < out[i].rows(); j++) res += square(out[i](j) - label[i](j));
        ans += sqrt(res / out[i].rows());
    }

    return ans / batch_sz;
}
// 分2类，交叉熵
double crossentropy_2(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++)
            res -= label[i](j) * mylog(out[i](j)) + (1. - label[i](j)) * mylog(1. - out[i](j));
    return res / batch_sz;
}
double crossentropy_k(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res -= label[i](j) * mylog(out[i](j));

    return res / batch_sz;
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
    return -res / batch_sz;
}
double chk_2(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    double res = 0;
    for (int i = 0; i < batch_sz; i++) {
        for (int j = 0; j < out[i].rows(); j++) res += (out[i](j) > 0.5) == (label[i](j) > 0.5);
    }
    return -res / batch_sz / out[0].rows();
}
//}}}
//{{{ Optimizer
struct optimizer {
    virtual void upd(mat &, const mat &) = 0;
};

// 一个类，用来维护每一个系数矩阵对应的额外信息，如动量之类的
template <const int N>
struct optimizer_holder : public optimizer {
    unordered_map<const mat *, mat> V[N];
    template <const int I>
    mat &get(const mat &x) {
        auto it = V[I].find(&x);
        if (it != V[I].end())
            return it->second;
        else
            return V[I][&x] = mat::Zero(x.rows(), x.cols());
    }
};

struct sgd : public optimizer_holder<0> {
    double lr, lambda;
    sgd(double lr = 0.01, double lambda = 0) : lr(lr), lambda(lambda) {}
    void upd(mat &w, const mat &gw) { w -= (gw + w * lambda) * lr; }
};
struct nesterov : public optimizer_holder<1> {
    double alpha, mu, lambda;
    nesterov(double alpha = 0.01, double mu = 0.9, double lambda = 0) : alpha(alpha), mu(mu), lambda(lambda) {}
    void upd(mat &w, const mat &gw) {
        // 原始版：
        // w'+=alpha*v
        // gw=grad(w')
        // v=alpha*v-gw*mu
        // w+=v
        // 修改版：
        // gw=grad(w)
        // w-=alpha*v;
        // v=alpha*b-gw*mu;
        // w+=v*(1+alpha);
        mat &v = get<0>(w);
        w -= mu * v;
        v = mu * v - (gw + lambda * w) * alpha;
        w += (1. + mu) * v;
    }
};

struct adam : public optimizer_holder<2> {
    double lr, rho1, rho2, eps;
    double mult1, mult2;
    adam(double lr = 0.001, double rho1 = 0.9, double rho2 = 0.999, double eps = 1e-6)
        : lr(lr), rho1(rho1), rho2(rho2), eps(eps), mult1(1), mult2(1) {}
    void upd(mat &w, const mat &gw) {
        mat &s = get<0>(w), &r = get<1>(w);
        mult1 *= rho1, mult2 *= rho2;
        s = s * rho1 + gw * (1. - rho1);
        r = r * rho2 + gw.cwiseAbs2() * (1. - rho2);
        w.array() -= lr * s.array() / (sqrt(r.array() / (1. - mult2) + eps)) / (1. - mult1);
    }
};

//}}}
//{{{ Layer
struct layer;
using layerp = shared_ptr<layer>;

// 一层神经元的基类
struct layer {
    string name;
    vec_batch out, grad;
    int batch_sz;
    bool train_mode;

    layer(const string &name) : name(name), batch_sz(0), train_mode(1) {}

    // 更改batch-size 和 trainmode 选项
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
    // 把输入向量运算后放到Z里
    virtual void forward(const vec_batch &in) {}
    // 根据输入向量，和答案对下一层神经元输入的偏导，算出答案对这一层神经元输入的偏导
    virtual void backward(const vec_batch &, const vec_batch &) = 0;

    virtual void clear_grad() {}
    // 返回这一层对应的系数，梯度矩阵的指针
    virtual void write(ostream &io) {}
    virtual void read(istream &io) {}
    // 以 rate 学习率梯度下降
    virtual void sgd(double) {}
    virtual void upd(optimizer &opt) {}
};

//}}}
//{{{ Activate Function
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
struct swish : public layer {
    swish() : layer("swish") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++)
            out[i] = in[i].unaryExpr([](double x) -> double { return x / (exp(-x) + 1); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](double x) -> double {
                double ex = std::exp(x);
                return ex * (x + ex + 1) / (ex + 1) / (ex + 1);
            }));
    }
};
struct mish : public layer {
    mish() : layer("mish") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++)
            out[i] = in[i].unaryExpr([](double x) -> double { return x * tanh(log(1 + exp(x))); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](double x) -> double {
                double ex = exp(x);
                double fm = (2 * ex + ex * ex + 2);
                return ex * (4 * (x + 1) + 4 * ex * ex + ex * ex * ex + ex * (4 * x + 6)) / fm / fm;
            }));
    }
};
struct softmax : public layer {
    softmax() : layer("softmax") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) {
            out[i] = exp(in[i].array() - in[i].maxCoeff());
            out[i] /= out[i].sum();
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) { assert(0); }
};
//}}}
//{{{ Layer to be trained
struct linear : public layer {
    const int in_sz, out_sz;
    // 系数，系数梯度
    mat weight, grad_weight;
    mat bias, grad_bias;
    linear(int in_sz, int out_sz)
        : layer("linear " + to_string(in_sz) + " -> " + to_string(out_sz))
        , in_sz(in_sz)
        , out_sz(out_sz)
        , weight(out_sz, in_sz)
        , grad_weight(out_sz, in_sz)
        , bias(out_sz, 1)
        , grad_bias(out_sz, 1) {
        bias.setZero();
        for (auto &i : weight.reshaped()) i = nd(0, 2. / in_sz);
    }
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = weight * in[i] + bias;
    }
    void clear_grad() {
        grad_weight.setZero();
        grad_bias.setZero();
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        // nx.in==W*this.in+B
        // dnx.in/din
        for (int i = 0; i < batch_sz; i++) {
            grad_bias += nxt_grad[i];
            grad_weight += nxt_grad[i] * in[i].transpose();
            grad[i] = weight.transpose() * nxt_grad[i];
        }
        grad_weight /= batch_sz;
        grad_bias /= batch_sz;
    }
    void upd(optimizer &opt) {
        opt.upd(weight, grad_weight);
        opt.upd(bias, grad_bias);
    }
    void write(ostream &io) {
        for (auto &i : weight.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : bias.reshaped()) io.write((char *)&i, sizeof(i));
    }
    void read(istream &io) {
        for (auto &i : weight.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : bias.reshaped()) io.read((char *)&i, sizeof(i));
    }
};
struct batchnorm : public layer {
    // 平均值，方差
    mat mean, running_mean, grad_mean;
    mat var, running_var, grad_var, inv_var;
    mat gama, grad_gama;
    mat beta, grad_beta;
    // 这两个用来辅助,inv记录1/sqrt(方差+eps)
    mat grad_normalized_x;
    const double momentum;
    batchnorm(int sz, double momentum = 0.9)
        : layer("batchnorm " + to_string(sz))
        , mean(sz, 1)
        , running_mean(sz, 1)
        , grad_mean(sz, 1)
        , var(sz, 1)
        , running_var(sz, 1)
        , grad_var(sz, 1)
        , inv_var(sz, 1)
        , gama(sz, 1)
        , grad_gama(sz, 1)
        , beta(sz, 1)
        , grad_beta(sz, 1)
        , momentum(momentum) {
        gama.setOnes();
        beta.setZero();
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
            // 使用无偏方差
            running_var = running_var * momentum + var * batch_sz / (batch_sz - 1) * (1 - momentum);
            /* running_var = running_var * momentum + var * (1 - momentum); */

            for (int i = 0; i < batch_sz; i++)
                out[i] = (in[i] - mean).array() * inv_var.array() * gama.array() + beta.array();
        } else {
            for (int i = 0; i < batch_sz; i++)
                out[i] =
                    (in[i] - running_mean).array() * rsqrt(running_var.array() + EPS) * gama.array() + beta.array();
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) {
            grad_normalized_x = nxt_grad[i].array() * gama.array();

            grad_var.array() += grad_normalized_x.array() * (in[i] - mean).array();
            grad_mean.array() += grad_normalized_x.array();

            grad[i] = grad_normalized_x.array() * inv_var.array();

            grad_beta.array() += nxt_grad[i].array();
            grad_gama.array() += nxt_grad[i].array() * (in[i] - mean).array() * inv_var.array();
        }
        grad_var = -0.5 * grad_var.array() * inv_var.array().cube();
        grad_mean = -grad_mean.array() * inv_var.array();
        for (int i = 0; i < batch_sz; i++)
            grad[i].array() += (grad_mean.array() + 2 * grad_var.array() * (in[i] - mean).array()) / batch_sz;
        grad_beta /= batch_sz;
        grad_gama /= batch_sz;
    }
    void clear_grad() {
        grad_gama.setZero();
        grad_beta.setZero();
        grad_mean.setZero();
        grad_var.setZero();
    }
    void upd(optimizer &opt) {
        opt.upd(beta, grad_beta);
        opt.upd(gama, grad_gama);
    }
    void write(ostream &io) {
        for (auto &i : gama.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : beta.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : running_mean.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : running_var.reshaped()) io.write((char *)&i, sizeof(i));
    }
    void read(istream &io) {
        for (auto &i : gama.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : beta.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : running_mean.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : running_var.reshaped()) io.read((char *)&i, sizeof(i));
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
    void upd(optimizer &opt, const batch &data) {
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->clear_grad();
        forward(data.first);
        backward(data.second);
        for (int i = 0; i < layer_sz; i++) layers[i]->upd(opt);
    }
    void writef(const string &f) {
        ofstream fout(f, ios::binary | ios::out);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->write(fout);
        fout.close();
    }
    void readf(const string &f) {
        ifstream fin(f, ios::binary | ios::in);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->read(fin);
        fin.close();
    }
};
//}}}
//{{{ Dataset
struct data_set {
    batch train, valid;
    data_set() {}
    data_set(const batch &all_data) {
        for (int i = 0; i < (int)all_data.first.size(); i++) {
            int rnd = ri(0, 6);
            if (rnd == 0) {
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

void upd(optimizer &opt, const data_set &data, layer_seq &net, int batch_sz, int epoch,
         function<double(const vec_batch &, const vec_batch &)> err_func, const string &save_file = "") {
    int t0 = clock();
    double tloss = 0, mult = 1, mn = INF;
    for (int i = 1; i <= epoch; i++) {
        auto tmp = data.get_train_batch(batch_sz);
        net.upd(opt, tmp);
        mult *= 0.9;
        tloss = tloss * 0.9 + err_func(net.layers.back()->out, tmp.second) * 0.1;
        if (i % 100 == 0) {
            cerr << "-------------------------" << endl;
            cerr << "Time elapse: " << (double)(clock() - t0) / CLOCKS_PER_SEC << endl;
            cerr << "Epoch: " << i << endl;
            cerr << "Loss: " << tloss / (1. - mult) << endl;
            if (i % 1000 == 0) {
                net.set_train_mode(0);
                double sum = 0;
                for (int j = 0; j < (int)data.valid.first.size(); j++) {
                    batch tmp = {{data.valid.first[j]}, {data.valid.second[j]}};
                    sum += err_func(net.forward(tmp.first), tmp.second);
                }
                net.set_train_mode(1);
                sum /= data.valid.first.size();
                cerr << "!! Error: " << sum << endl;
                if (sum < mn && save_file != "") {
                    cerr << "Saved" << endl;
                    mn = sum;
                    net.writef(save_file);
                }
            }
        }
    }
}

//}}}

#endif
