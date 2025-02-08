# MLX built with Zig

This project builds MLX with Zig instead of the CMake build system. Meaning that these are not Zig language bindings to MLX. Note that not all features from the Cmake build are yet ported over to Zig. 

To explain more further how this is setup, let's look at how the Swift bindings work. [MLX](https://github.com/ml-explore/mlx) is mainly written C++ and on top of that we have [MLX-C](https://github.com/ml-explore/mlx-c), a small C wrapper around MLX. And finally [MLX-Swift](https://github.com/ml-explore/mlx-swift/) which creates the C to Swift language bindings. The reason the C wrapper is needed in between is because most languages adhere to the C ABI, so it's the natural layer for glueing things together.

You can make Zig bindings following a similar pattern, but since Zig is also a C/C++ compiler, we can integrate even further. We can build both MLX and MLX-C using Zig, which allows us to import them in the same way as code written in Zig. This also means that you get all the control of building MLX from the ground up. For example if you don't need support for GGUF or Safetensors? Just leave them out from the build.

In the chart this repo is the leftmost in the lower flow. So it's not strickly fork of MLX nor is this the repo for Zig bindings to MLX. You can find the [Zig-build-mlx-C repo here]() and a repo exporing an example of language bindings [here](zig-mlx-example-bindings) 

![mlx-chart](https://github.com/erikkaum/zig-build-mlx/blob/main/assets/chart-mlx.png?raw=true)


## Supported features

- [x] CPU tests
- [ ] Metal tests
- [ ] Examaples
- [ ] Benchmarks
- [ ] Python bindings
- [ ] Metal
- [x] Build CPU
- [ ] Metal debug
- [ ] Enable x68 Mac
- [ ] GGUF
- [ ] Safetensors
- [ ] Metal JIT
- [ ] Shared libs

Besides these there are a bunch of TODOs in the `build.zig` file to double check that we're actually porting the functionality from CMake correctly. Also the build could probably be faster, I doubt things are done optimally now.

## Usage

Create a `build.zig.zon` like so:

```zig
.{
    .name = "my-project",
    .version = "0.0.0",
    .dependencies = .{
        .mlx = .{
            .url = "https://github.com/erikkaum/zig-build-mlx/archive/<git-ref-here>.tar.gz",
            .hash = "",
        },
    },
}
```

And in your `build.zig`:

```zig
const mlx = b.dependency("mlx", .{ .target = target, .optimize = optimize });
exe.linkLibrary(mlx.artifact("mlx"));
```
