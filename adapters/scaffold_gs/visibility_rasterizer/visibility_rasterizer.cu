#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
constexpr int kBlockSize = kBlockX * kBlockY;
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kStopTransmittance = 0.0001f;

__device__ __forceinline__ float contribution(float2 center, float4 conic_opacity, float2 pixel, float transmittance) {
  const float dx = center.x - pixel.x;
  const float dy = center.y - pixel.y;
  const float power = -0.5f * (conic_opacity.x * dx * dx + conic_opacity.z * dy * dy) - conic_opacity.y * dx * dy;
  if (power > 0.0f) return 0.0f;
  const float alpha = fminf(0.99f, conic_opacity.w * expf(power));
  return alpha < kMinAlpha ? 0.0f : transmittance * alpha;
}

__global__ void contribution_stats_kernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float2* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    int width,
    int height,
    float* __restrict__ foreground_mass,
    float* __restrict__ max_contribution,
    int32_t* __restrict__ contributor_count,
    int32_t* __restrict__ dominant_point_index) {
  const uint2 tile_origin = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
  const uint2 pixel = {tile_origin.x + threadIdx.x, tile_origin.y + threadIdx.y};
  const bool inside = pixel.x < width && pixel.y < height;
  const int pixel_id = pixel.y * width + pixel.x;
  const int tile_count_x = (width + kBlockX - 1) / kBlockX;
  const uint2 range = ranges[blockIdx.y * tile_count_x + blockIdx.x];
  const int work = static_cast<int>(range.y - range.x);
  const int rounds = (work + kBlockSize - 1) / kBlockSize;

  __shared__ uint32_t shared_ids[kBlockSize];
  __shared__ float2 shared_means[kBlockSize];
  __shared__ float4 shared_conics[kBlockSize];

  float transmittance = 1.0f;
  float max_r = 0.0f;
  int count = 0;
  int32_t dominant = -1;
  bool done = !inside;
  const float2 pixel_f = {static_cast<float>(pixel.x), static_cast<float>(pixel.y)};
  for (int round = 0; round < rounds; ++round) {
    const int item = round * kBlockSize + threadIdx.y * kBlockX + threadIdx.x;
    if (range.x + item < range.y) {
      const uint32_t id = point_list[range.x + item];
      shared_ids[threadIdx.y * kBlockX + threadIdx.x] = id;
      shared_means[threadIdx.y * kBlockX + threadIdx.x] = means2d[id];
      shared_conics[threadIdx.y * kBlockX + threadIdx.x] = conic_opacity[id];
    }
    __syncthreads();
    for (int j = 0; !done && j < min(kBlockSize, work - round * kBlockSize); ++j) {
      const float r = contribution(shared_means[j], shared_conics[j], pixel_f, transmittance);
      if (r == 0.0f) continue;
      const float alpha = r / transmittance;
      const float next_t = transmittance * (1.0f - alpha);
      if (next_t < kStopTransmittance) {
        done = true;
        continue;
      }
      transmittance = next_t;
      if (r > max_r) {
        max_r = r;
        dominant = static_cast<int32_t>(shared_ids[j]);
      }
      ++count;
    }
    __syncthreads();
  }
  if (inside) {
    foreground_mass[pixel_id] = 1.0f - transmittance;
    max_contribution[pixel_id] = max_r;
    contributor_count[pixel_id] = count;
    dominant_point_index[pixel_id] = dominant;
  }
}

__global__ void semantic_forward_kernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float2* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ probabilities,
    int width,
    int height,
    int gate_mode,
    float threshold,
    const float* __restrict__ foreground_mass,
    const float* __restrict__ max_contribution,
    float* __restrict__ numerator,
    int32_t* __restrict__ retained_count,
    float* __restrict__ retained_mass) {
  const uint2 tile_origin = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
  const uint2 pixel = {tile_origin.x + threadIdx.x, tile_origin.y + threadIdx.y};
  const bool inside = pixel.x < width && pixel.y < height;
  const int pixel_id = pixel.y * width + pixel.x;
  const int tile_count_x = (width + kBlockX - 1) / kBlockX;
  const uint2 range = ranges[blockIdx.y * tile_count_x + blockIdx.x];
  const int work = static_cast<int>(range.y - range.x);
  const int rounds = (work + kBlockSize - 1) / kBlockSize;

  __shared__ uint32_t shared_ids[kBlockSize];
  __shared__ float2 shared_means[kBlockSize];
  __shared__ float4 shared_conics[kBlockSize];

  float transmittance = 1.0f;
  float prefix = 0.0f;
  float channels[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
  float kept = 0.0f;
  int kept_count = 0;
  bool done = !inside;
  const float2 pixel_f = {static_cast<float>(pixel.x), static_cast<float>(pixel.y)};
  const float total = inside ? foreground_mass[pixel_id] : 0.0f;
  const float peak = inside ? max_contribution[pixel_id] : 0.0f;
  const float cumulative_limit = threshold * total;
  for (int round = 0; round < rounds; ++round) {
    const int item = round * kBlockSize + threadIdx.y * kBlockX + threadIdx.x;
    if (range.x + item < range.y) {
      const uint32_t id = point_list[range.x + item];
      shared_ids[threadIdx.y * kBlockX + threadIdx.x] = id;
      shared_means[threadIdx.y * kBlockX + threadIdx.x] = means2d[id];
      shared_conics[threadIdx.y * kBlockX + threadIdx.x] = conic_opacity[id];
    }
    __syncthreads();
    for (int j = 0; !done && j < min(kBlockSize, work - round * kBlockSize); ++j) {
      const float r = contribution(shared_means[j], shared_conics[j], pixel_f, transmittance);
      if (r == 0.0f) continue;
      const float alpha = r / transmittance;
      const float next_t = transmittance * (1.0f - alpha);
      if (next_t < kStopTransmittance) {
        done = true;
        continue;
      }
      const bool keep = gate_mode == 1 ? prefix < cumulative_limit : r >= threshold * peak;
      if (keep) {
        const uint32_t id = shared_ids[j];
        const float* values = probabilities + id * 5;
        #pragma unroll
        for (int channel = 0; channel < 5; ++channel) channels[channel] += values[channel] * r;
        kept += r;
        ++kept_count;
      }
      prefix += r;
      transmittance = next_t;
    }
    __syncthreads();
  }
  if (inside) {
    channels[0] += transmittance;
    #pragma unroll
    for (int channel = 0; channel < 5; ++channel) numerator[channel * height * width + pixel_id] = channels[channel];
    retained_count[pixel_id] = kept_count;
    retained_mass[pixel_id] = kept;
  }
}

__global__ void semantic_backward_kernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float2* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ grad_numerator,
    const float* __restrict__ foreground_mass,
    const float* __restrict__ max_contribution,
    int width,
    int height,
    int gate_mode,
    float threshold,
    float* __restrict__ grad_probabilities) {
  const uint2 tile_origin = {blockIdx.x * kBlockX, blockIdx.y * kBlockY};
  const uint2 pixel = {tile_origin.x + threadIdx.x, tile_origin.y + threadIdx.y};
  const bool inside = pixel.x < width && pixel.y < height;
  const int pixel_id = pixel.y * width + pixel.x;
  const int tile_count_x = (width + kBlockX - 1) / kBlockX;
  const uint2 range = ranges[blockIdx.y * tile_count_x + blockIdx.x];
  const int work = static_cast<int>(range.y - range.x);
  const int rounds = (work + kBlockSize - 1) / kBlockSize;

  __shared__ uint32_t shared_ids[kBlockSize];
  __shared__ float2 shared_means[kBlockSize];
  __shared__ float4 shared_conics[kBlockSize];

  float transmittance = 1.0f;
  float prefix = 0.0f;
  bool done = !inside;
  const float2 pixel_f = {static_cast<float>(pixel.x), static_cast<float>(pixel.y)};
  const float total = inside ? foreground_mass[pixel_id] : 0.0f;
  const float peak = inside ? max_contribution[pixel_id] : 0.0f;
  const float cumulative_limit = threshold * total;
  for (int round = 0; round < rounds; ++round) {
    const int item = round * kBlockSize + threadIdx.y * kBlockX + threadIdx.x;
    if (range.x + item < range.y) {
      const uint32_t id = point_list[range.x + item];
      shared_ids[threadIdx.y * kBlockX + threadIdx.x] = id;
      shared_means[threadIdx.y * kBlockX + threadIdx.x] = means2d[id];
      shared_conics[threadIdx.y * kBlockX + threadIdx.x] = conic_opacity[id];
    }
    __syncthreads();
    for (int j = 0; !done && j < min(kBlockSize, work - round * kBlockSize); ++j) {
      const float r = contribution(shared_means[j], shared_conics[j], pixel_f, transmittance);
      if (r == 0.0f) continue;
      const float alpha = r / transmittance;
      const float next_t = transmittance * (1.0f - alpha);
      if (next_t < kStopTransmittance) {
        done = true;
        continue;
      }
      const bool keep = gate_mode == 1 ? prefix < cumulative_limit : r >= threshold * peak;
      if (keep) {
        const uint32_t id = shared_ids[j];
        #pragma unroll
        for (int channel = 0; channel < 5; ++channel) {
          atomicAdd(grad_probabilities + id * 5 + channel, r * grad_numerator[channel * height * width + pixel_id]);
        }
      }
      prefix += r;
      transmittance = next_t;
    }
    __syncthreads();
  }
}

}  // namespace

std::vector<torch::Tensor> visibility_forward_cuda(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor probabilities,
    int64_t height,
    int64_t width,
    int64_t gate_mode,
    double threshold) {
  const auto float_options = probabilities.options().dtype(torch::kFloat32);
  const auto int_options = probabilities.options().dtype(torch::kInt32);
  auto foreground = torch::zeros({height, width}, float_options);
  auto maximum = torch::zeros({height, width}, float_options);
  auto count = torch::zeros({height, width}, int_options);
  auto dominant = torch::full({height, width}, -1, int_options);
  auto numerator = torch::zeros({5, height, width}, float_options);
  auto retained_count = torch::zeros({height, width}, int_options);
  auto retained_mass = torch::zeros({height, width}, float_options);
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY);
  contribution_stats_kernel<<<grid, block>>>(
      reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(point_list.data_ptr<int32_t>()),
      reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
      static_cast<int>(width), static_cast<int>(height), foreground.data_ptr<float>(), maximum.data_ptr<float>(), count.data_ptr<int32_t>(), dominant.data_ptr<int32_t>());
  semantic_forward_kernel<<<grid, block>>>(
      reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(point_list.data_ptr<int32_t>()),
      reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()), probabilities.data_ptr<float>(),
      static_cast<int>(width), static_cast<int>(height), static_cast<int>(gate_mode), static_cast<float>(threshold),
      foreground.data_ptr<float>(), maximum.data_ptr<float>(), numerator.data_ptr<float>(), retained_count.data_ptr<int32_t>(), retained_mass.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {numerator, foreground, maximum, count, retained_count, retained_mass};
}

std::vector<torch::Tensor> visibility_diagnostic_stats_cuda(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    int64_t height,
    int64_t width) {
  const auto float_options = means2d.options().dtype(torch::kFloat32);
  const auto int_options = means2d.options().dtype(torch::kInt32);
  auto foreground = torch::zeros({height, width}, float_options);
  auto maximum = torch::zeros({height, width}, float_options);
  auto count = torch::zeros({height, width}, int_options);
  auto dominant = torch::full({height, width}, -1, int_options);
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY);
  contribution_stats_kernel<<<grid, block>>>(
      reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(point_list.data_ptr<int32_t>()),
      reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
      static_cast<int>(width), static_cast<int>(height), foreground.data_ptr<float>(), maximum.data_ptr<float>(), count.data_ptr<int32_t>(), dominant.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {foreground, maximum, count, dominant};
}

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
    double threshold) {
  auto grad = torch::zeros({point_count, 5}, grad_numerator.options().dtype(torch::kFloat32));
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY);
  semantic_backward_kernel<<<grid, block>>>(
      reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(point_list.data_ptr<int32_t>()),
      reinterpret_cast<const float2*>(means2d.data_ptr<float>()),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()), grad_numerator.data_ptr<float>(),
      foreground_mass.data_ptr<float>(), max_contribution.data_ptr<float>(),
      static_cast<int>(width), static_cast<int>(height), static_cast<int>(gate_mode), static_cast<float>(threshold), grad.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad;
}
