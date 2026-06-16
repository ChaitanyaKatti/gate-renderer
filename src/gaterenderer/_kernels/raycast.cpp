#include <torch/extension.h>

// Forward declaration
void launch_raycast_kernel(torch::Tensor rays_cam, torch::Tensor c2w, torch::Tensor quads_p, torch::Tensor output_buf, int W, int H);

void raycast(torch::Tensor rays_cam, torch::Tensor c2w, torch::Tensor quads_p, torch::Tensor output_buf, int W, int H) {
    launch_raycast_kernel(rays_cam, c2w, quads_p, output_buf, W, H);
}
