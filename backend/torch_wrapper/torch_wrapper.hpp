#include <torch/torch.h>
#include <vector>

using namespace std;

typedef at::Tensor act_op(const at::Tensor&);

template <int nn_inputs, int nn_outputs, int nn_hidden_neurons, int nn_hidden_layers, act_op hidden_act, act_op output_act> struct FeedForwardNet : torch::nn::Module {
public:
  vector<shared_ptr<torch::nn::LinearImpl>> layers;

  FeedForwardNet() {
    layers.push_back(register_module("input_layer", torch::nn::Linear(nn_inputs, nn_hidden_neurons)));
    for (int i = 0; i < nn_hidden_layers - 1; i++)
      layers.push_back(register_module("hidden_layer_" + to_string(i + 1),
                                      torch::nn::Linear(nn_hidden_neurons, nn_hidden_neurons)));
    layers.push_back(register_module("output_layer", torch::nn::Linear(nn_hidden_neurons, nn_outputs)));
  }

  void update_coeffs(adouble coeffs[nn_total_coeffs]) {
    float coeffs_[nn_total_coeffs];
    for (int i = 0; i < nn_total_coeffs; i++)
      coeffs_[i] = coeffs[i].val;

    size_t num_copied = 0;
    for (auto& layer : layers)
      num_copied += copy_coeffs(layer, &coeffs_[num_copied]);

    assert(num_copied == nn_total_coeffs);
  }

  void update_coeffs(array<adouble, num_inputs>& coeffs) {
    adouble coeffs_[coeffs.size()];
    for (int i = 0; i < coeffs.size(); i++)
      coeffs_[i] = coeffs[i];
    update_coeffs(coeffs_);
  }

  template<typename T>
  size_t copy_coeffs(T& layer, float* src) {
    layer->weight = torch::from_blob(src, layer->weight.sizes(),
                                     torch::TensorOptions(torch::kFloat32)).clone();
    layer->weight.requires_grad_(true);

    layer->bias = torch::from_blob(&src[layer->weight.numel()], layer->bias.sizes(),
                                         torch::TensorOptions(torch::kFloat32)).clone();
    layer->bias.requires_grad_(true);

    return layer->weight.numel() + layer->bias.numel();
  }

  template<typename T>
  size_t copy_derivs(float *dst, T& layer) {
    size_t n_weight = layer->weight.grad().numel();
    size_t n_bias = layer->bias.grad().numel();

    float* raw_derivs = (float *)layer->weight.grad().data_ptr();
    for (size_t i = 0; i < n_weight; i++)
      dst[i] = raw_derivs[i];

    raw_derivs = (float *)layer->bias.grad().data_ptr();
    for (size_t i = 0; i < n_bias; i++) 
      dst[n_weight + i] = raw_derivs[i];

    return n_weight + n_bias;
  }

  void aforward(adouble x[nn_inputs], adouble y[nn_outputs]) {
    float data[nn_inputs];
    for (int i = 0; i < nn_inputs; i++)
      data[i] = x[i].get_val();

    torch::Tensor x_ = torch::from_blob(data, {1, nn_inputs}, torch::TensorOptions(torch::kFloat32)).clone();
    torch::Tensor y_ = forward(x_);

    float *raw_output = (float *)y_.data_ptr();

    for (int i = 0; i < nn_outputs; i++) {
      y[i] = raw_output[i];
      y_[0][i].backward({}, i < nn_outputs - 1); // don't discard tape before we've visited all outputs

      float derivs[nn_total_coeffs];
      size_t num_copied = 0;

      for (auto &layer : layers)
        num_copied += copy_derivs(&derivs[num_copied], layer);

      assert(num_copied == nn_total_coeffs);

      float *raw_input_derivs = (float *)x_.grad().data_ptr();

      for (int j = 0; j < nn_total_coeffs; j++) {
        double deriv = derivs[j];
        for (int k = 0; k < nn_inputs; k++)
          deriv += x[k].get_tang(j) * raw_input_derivs[k];
        y[i].set_tang(j, deriv); 
      }
    }

    // need to reset gradients manually since torch doesn't track our custom coeffs
    // so we can't just use zero_grad()
    for (auto &layer : layers) {
      layer->weight.grad().zero_();
      layer->bias.grad().zero_();
    }

    //for (int i = 0; i < nn_outputs; i++) {
    //  printf("y[%d]: %.4e\n", i, y[i].val);
    //  for (int j = 0; j < nn_total_coeffs; j++)
    //    printf("  deriv: %.4e\n", y[i].get_tang(j));
    //}
  }

  torch::Tensor forward(torch::Tensor input) {
    input.requires_grad_(true);

    auto r = hidden_act(layers.front()->forward(input));

    for (int i = 1; i < layers.size() - 1; i++)
      r = hidden_act(layers[i]->forward(r));

    return output_act(layers.back()->forward(r));
  }
};
