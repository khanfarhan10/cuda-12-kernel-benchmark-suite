#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor cuda_vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor cuda_tiled_matmul(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA 12.3+ custom kernel benchmarks — PyTorch C++ extension";
    m.def("vector_add",    &cuda_vector_add,    "Custom CUDA vector addition");
    m.def("tiled_matmul",  &cuda_tiled_matmul,  "Custom CUDA tiled matrix multiplication");
}
