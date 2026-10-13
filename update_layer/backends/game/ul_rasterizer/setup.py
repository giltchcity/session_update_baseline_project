# FlashSplat rasterizer (GaME submodule @1c971d6, from 3DGS by Inria GRAPHDECO; LICENSE.md) with a forward-only
# probe for update_layer: label votes from depth-agreeing Gaussians and the median depth. Build:
#   cd update_layer/backends/game/ul_rasterizer && <gs-cu128 python> -m pip install --no-build-isolation .
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
here = os.path.dirname(os.path.abspath(__file__))
setup(
    name="ul_flashsplat_rasterization",
    packages=["ul_flashsplat_rasterization"],
    ext_modules=[CUDAExtension(
        name="ul_flashsplat_rasterization._C",
        sources=["cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu", "cuda_rasterizer/backward.cu",
                 "rasterize_points.cu", "ext.cpp"],
        # glm (header-only) from GaME's checkout of the same rasterizer, not vendored here
        extra_compile_args={"nvcc": ["-I/home/jixian/Desktop/FT/baselines/GaME/submodules/flashsplat-rasterization/third_party/glm/"]})],
    cmdclass={"build_ext": BuildExtension})
