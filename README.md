# CudaRw

CudaRw is a dual read-write lock backed by a host buffer and a gpu buffer that lazily keeps them in sync

CudaRw depends on [cudaz](https://github.com/akhildevelops/cudaz) to provide CUDA functionality

## Building

Build CudaRw with `zig build`

## Depending

Add CudaRw as a dependency with `zig fetch --save git+https://github.com/JackDyre/cudarw`
