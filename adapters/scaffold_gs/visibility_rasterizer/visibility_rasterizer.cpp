#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> visibility_forward_cuda(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor probabilities,
    int64_t height,
    int64_t width,
    int64_t gate_mode,
    double threshold);

std::vector<torch::Tensor> visibility_diagnostic_stats_cuda(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    int64_t height,
    int64_t width);

torch::Tensor visibility_backward_cuda(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor grad_numerator,
    torch::Tensor foreground_mass,
    torch::Tensor max_contribution,
    int64_t height,
    int64_t width,
    int64_t point_count,
    int64_t gate_mode,
    double threshold);

namespace {

void check_inputs(
    const torch::Tensor& ranges,
    const torch::Tensor& point_list,
    const torch::Tensor& means2d,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& probabilities,
    int64_t height,
    int64_t width,
    int64_t gate_mode,
    double threshold) {
  TORCH_CHECK(ranges.is_cuda() && point_list.is_cuda() && means2d.is_cuda() && conic_opacity.is_cuda() && probabilities.is_cuda(), "visibility inputs must be CUDA tensors");
  TORCH_CHECK(ranges.is_contiguous() && point_list.is_contiguous() && means2d.is_contiguous() && conic_opacity.is_contiguous() && probabilities.is_contiguous(), "visibility inputs must be contiguous");
  TORCH_CHECK(ranges.scalar_type() == torch::kInt32 && ranges.dim() == 2 && ranges.size(1) == 2, "ranges must be int32 [tiles, 2]");
  TORCH_CHECK(point_list.scalar_type() == torch::kInt32 && point_list.dim() == 1, "point_list must be int32 [instances]");
  TORCH_CHECK(means2d.scalar_type() == torch::kFloat32 && means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be float32 [points, 2]");
  TORCH_CHECK(conic_opacity.scalar_type() == torch::kFloat32 && conic_opacity.dim() == 2 && conic_opacity.size(1) == 4, "conic_opacity must be float32 [points, 4]");
  TORCH_CHECK(probabilities.scalar_type() == torch::kFloat32 && probabilities.dim() == 2 && probabilities.size(1) == 5, "probabilities must be float32 [points, 5]");
  TORCH_CHECK(means2d.size(0) == conic_opacity.size(0) && means2d.size(0) == probabilities.size(0), "point tensor lengths must match");
  TORCH_CHECK(height > 0 && width > 0, "image dimensions must be positive");
  TORCH_CHECK(gate_mode == 1 || gate_mode == 2, "gate_mode must be 1 (cumulative) or 2 (relative-max)");
  TORCH_CHECK(threshold > 0.0 && threshold <= 1.0, "threshold must be in (0, 1]");
}

}  // namespace

std::vector<torch::Tensor> visibility_forward(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor probabilities,
    int64_t height,
    int64_t width,
    int64_t gate_mode,
    double threshold) {
  check_inputs(ranges, point_list, means2d, conic_opacity, probabilities, height, width, gate_mode, threshold);
  return visibility_forward_cuda(ranges, point_list, means2d, conic_opacity, probabilities, height, width, gate_mode, threshold);
}

torch::Tensor visibility_backward(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor grad_numerator,
    torch::Tensor foreground_mass,
    torch::Tensor max_contribution,
    int64_t height,
    int64_t width,
    int64_t point_count,
    int64_t gate_mode,
    double threshold) {
  TORCH_CHECK(grad_numerator.is_cuda() && foreground_mass.is_cuda() && max_contribution.is_cuda(), "backward inputs must be CUDA tensors");
  TORCH_CHECK(grad_numerator.scalar_type() == torch::kFloat32 && grad_numerator.sizes() == torch::IntArrayRef({5, height, width}), "grad_numerator must be float32 [5, H, W]");
  TORCH_CHECK(foreground_mass.scalar_type() == torch::kFloat32 && foreground_mass.sizes() == torch::IntArrayRef({height, width}), "foreground_mass must be float32 [H, W]");
  TORCH_CHECK(max_contribution.scalar_type() == torch::kFloat32 && max_contribution.sizes() == torch::IntArrayRef({height, width}), "max_contribution must be float32 [H, W]");
  TORCH_CHECK(point_count > 0, "point_count must be positive");
  return visibility_backward_cuda(ranges, point_list, means2d, conic_opacity, grad_numerator, foreground_mass, max_contribution, height, width, point_count, gate_mode, threshold);
}

std::vector<torch::Tensor> visibility_diagnostic_stats(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(ranges.is_cuda() && point_list.is_cuda() && means2d.is_cuda() && conic_opacity.is_cuda(), "visibility diagnostic inputs must be CUDA tensors");
  TORCH_CHECK(ranges.is_contiguous() && point_list.is_contiguous() && means2d.is_contiguous() && conic_opacity.is_contiguous(), "visibility diagnostic inputs must be contiguous");
  TORCH_CHECK(ranges.scalar_type() == torch::kInt32 && ranges.dim() == 2 && ranges.size(1) == 2, "ranges must be int32 [tiles, 2]");
  TORCH_CHECK(point_list.scalar_type() == torch::kInt32 && point_list.dim() == 1, "point_list must be int32 [instances]");
  TORCH_CHECK(means2d.scalar_type() == torch::kFloat32 && means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be float32 [points, 2]");
  TORCH_CHECK(conic_opacity.scalar_type() == torch::kFloat32 && conic_opacity.dim() == 2 && conic_opacity.size(1) == 4, "conic_opacity must be float32 [points, 4]");
  TORCH_CHECK(means2d.size(0) == conic_opacity.size(0), "point tensor lengths must match");
  TORCH_CHECK(height > 0 && width > 0, "image dimensions must be positive");
  return visibility_diagnostic_stats_cuda(ranges, point_list, means2d, conic_opacity, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &visibility_forward, "Contribution-gated semantic forward (CUDA)");
  m.def("backward", &visibility_backward, "Contribution-gated semantic backward (CUDA)");
  m.def("diagnostic_stats", &visibility_diagnostic_stats, "Frozen per-pixel contribution diagnostics (CUDA)");
}
