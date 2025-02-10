const std = @import("std");

// Build options equivalent to CMake options
const BuildOptions = struct {
    build_tests: bool,
    build_examples: bool,
    build_benchmarks: bool,
    build_python_bindings: bool,
    build_metal: bool,
    build_cpu: bool,
    metal_debug: bool,
    enable_x64_mac: bool,
    build_gguf: bool,
    build_safetensors: bool,
    metal_jit: bool,
    shared_libs: bool,
    install_message: bool,
    version: []const u8,

    fn fromOptions(b: *std.Build) BuildOptions {
        return .{
            // TODO check these defaults and align with MLX
            .build_tests = b.option(bool, "build-tests", "Build tests for mlx") orelse true,
            .build_examples = b.option(bool, "build-examples", "Build examples for mlx") orelse false,
            .build_benchmarks = b.option(bool, "build-benchmarks", "Build benchmarks for mlx") orelse false,
            .build_python_bindings = b.option(bool, "build-python", "Build python bindings for mlx") orelse false,
            .build_metal = b.option(bool, "build-metal", "Build metal backend") orelse false,
            .build_cpu = b.option(bool, "build-cpu", "Build cpu backend") orelse true,
            .metal_debug = b.option(bool, "metal-debug", "Enhance metal debug workflow") orelse false,
            .enable_x64_mac = b.option(bool, "enable-x64-mac", "Enable building for x64 macOS") orelse false,
            .build_gguf = b.option(bool, "build-gguf", "Include support for GGUF format") orelse true,
            .build_safetensors = b.option(bool, "build-safetensors", "Include support for safetensors format") orelse false,
            .metal_jit = b.option(bool, "metal-jit", "Use JIT compilation for Metal kernels") orelse false,
            .shared_libs = b.option(bool, "shared-libs", "Build mlx as a shared library") orelse false,
            .install_message = b.option(bool, "install-message", "Show installation messages") orelse false,
            .version = b.option([]const u8, "version", "MLX version") orelse "0.21.0",
        };
    }
};

// Add compiler flags as constants
const CPP_FLAGS = [_][]const u8{
    "-std=c++17",
    "-fPIC",
    "-DACCELERATE_NEW_LAPACK", // TODO this should be set conditionally
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    // "-frtti",
    "-fexceptions",
};

// Add compiler flags as constants
const C_FLAGS = [_][]const u8{
    "-fPIC",
    "-DACCELERATE_NEW_LAPACK", // TODO this should be set conditionally
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    // "-frtti",
    "-fexceptions",
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    // TODO: standard optimizer gives a trap exit and null pointer bug, don't know exactly why but I have a hunch
    // const optimize = b.standardOptimizeOption(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;

    // Parse build options using the new fromOptions helper
    const options = BuildOptions.fromOptions(b);

    // Initialize dependencies
    const deps = try Dependencies.init(b, options, target, optimize);

    // Original MLX, let's not call it "mlx" since that could be easy to confuse with the mlx lib we're now building
    const og_mlx = b.dependency("mlx", .{
        .target = target,
        .optimize = optimize,
    });

    // Zig built MLX
    const lib = b.addStaticLibrary(.{
        .name = "mlx",
        .target = target,
        .optimize = optimize,
    });

    lib.addIncludePath(deps.fmt.path("include"));

    // TODO this gets all headers, not sure we need all e.g. for metal kernels etc
    lib.installHeadersDirectory(og_mlx.path("."), ".", .{});
    lib.addIncludePath(og_mlx.path("."));
    lib.linkLibCpp();

    // Add core sources
    lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &core_sources, .flags = &CPP_FLAGS });

    // TODO check that the files included for safetensors and ggufs are really correct
    if (options.build_safetensors) {
        lib.addCSourceFile(.{
            .file = b.path("upstream/mlx/mlx/io/safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    } else {
        lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/no_safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    }

    if (options.build_gguf) {
        const gguf_dep = deps.gguflib.?;

        lib.addIncludePath(gguf_dep.path("."));

        const gguflib_lib = b.addStaticLibrary(.{
            .name = "gguflib",
            .target = target,
            .optimize = optimize,
        });

        const gguflib_sources = [_][]const u8{
            "fp16.c",
            "gguflib.c",
        };

        gguflib_lib.addCSourceFiles(.{
            .root = gguf_dep.path("."),
            .files = &gguflib_sources,
            .flags = &C_FLAGS,
        });

        lib.linkLibrary(gguflib_lib);

        const gguf_sources = [_][]const u8{
            "io/gguf.cpp",
            "io/gguf_quants.cpp",
        };
        lib.addCSourceFiles(.{
            .root = og_mlx.path("mlx"),
            .files = &gguf_sources,
            .flags = &C_FLAGS,
        });
    } else {
        lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/no_gguf.cpp"),
            .flags = &CPP_FLAGS,
        });
    }

    // TODO check if this really works
    const mpi_result = try checkMPI();
    if (mpi_result.found) {
        if (mpi_result.include_path) |path| {
            lib.addIncludePath(b.path(path));
        }

        lib.addCSourceFiles(.{
            .files = &distributed_sources,
            .flags = &CPP_FLAGS,
        });
    }

    const is_darwin = target.result.isDarwin();
    const is_arm = std.Target.Cpu.Arch.isAARCH64(target.result.cpu.arch);
    const is_x86_64 = std.Target.Cpu.Arch.isX86(target.result.cpu.arch);

    try build_preamble(b, lib, og_mlx, is_darwin);

    // Validate system requirements
    if (is_darwin and is_x86_64 and !options.enable_x64_mac) {
        @panic("Building for x86_64 on macOS is not supported. If you are on an Apple silicon system, check the build documentation.");
    }

    // Check SDK version for Metal
    if (is_darwin and options.build_metal) {
        const sdk_version = try checkMacOSSDKVersion();
        if (sdk_version < 14.0) {
            @panic("MLX requires macOS SDK >= 14.0 to be built with MLX_BUILD_METAL=ON");
        }
    }

    // Metal support (Darwin only)
    if (options.build_metal) {
        lib.linkFramework("Metal");
        lib.linkFramework("Foundation");
        lib.linkFramework("QuartzCore");

        lib.addCSourceFiles(.{ .files = &metal_sources, .flags = &CPP_FLAGS });

        const metal_builder = try MetalKernelBuilder.init(b, target, options);
        try metal_builder.buildCoreKernels();

        if (options.metal_jit) {
            try metal_builder.buildJitKernels();
        } else {
            try metal_builder.buildNonJitKernels();
        }

        try metal_builder.buildMetallib();

        // Add Metal-specific compile definition
        lib.defineCMacro("METAL_PATH", try std.fmt.allocPrint(
            b.allocator,
            "{s}/mlx.metallib",
            .{metal_builder.metal_path},
        ));

        // Add conditional source files based on JIT
        if (options.metal_jit) {
            lib.addCSourceFile(.{
                .file = b.path("upstream/mlx/mlx/backend/metal/jit_kernels.cpp"),
                .flags = &CPP_FLAGS,
            });
        } else {
            lib.addCSourceFile(.{
                .file = b.path("upstream/mlx/mlx/backend/metal/nojit_kernels.cpp"),
                .flags = &CPP_FLAGS,
            });
        }
    } else {
        lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &no_metal_sources, .flags = &CPP_FLAGS });
    }

    if (options.build_cpu) {
        lib.addCSourceFiles(.{
            .root = og_mlx.path("mlx"),
            .files = &common_sources,
            .flags = &CPP_FLAGS,
        });

        // Add IOS-specific sources
        if (target.query.os_tag == std.Target.Os.Tag.ios) { // TODO check if this is way of confirming iOS
            lib.addCSourceFile(.{
                .file = b.path("upstream/mlx/mlx/backend/common/compiled_nocpu.cpp"),
                .flags = &CPP_FLAGS,
            });
        } else {
            lib.addCSourceFile(.{
                .file = og_mlx.path("mlx/backend/common/compiled_cpu.cpp"),
                .flags = &CPP_FLAGS,
            });
        }

        // Add default primitives when not using Accelerate
        if (options.build_cpu and !is_darwin) {
            lib.addCSourceFile(.{
                .file = b.path("upstream/mlx/mlx/backend/common/default_primitives.cpp"),
                .flags = &CPP_FLAGS,
            });
        }
    } else {
        lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &no_cpu_sources, .flags = &CPP_FLAGS });
    }

    // Link with system libraries
    if (is_darwin) {
        if (is_arm) {
            lib.linkFramework("Accelerate");
            lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &accelerate_sources, .flags = &CPP_FLAGS });
        }
    } else {
        // TODO check if these are correct
        lib.linkSystemLibrary("stdc++");
        lib.linkSystemLibrary("lapack");
        lib.linkSystemLibrary("blas");
    }

    // Link dependencies
    // if (deps.gguflib) |gguf| {
    //     lib.linkLibrary(gguf.artifact("gguf"));
    // }
    // if (deps.json) |json| {
    //     lib.addIncludePath(json.path(b.pathJoin(&.{ "single_include", "nlohmann" })));
    // }

    // Install
    b.installArtifact(lib);

    if (options.build_tests) {
        const tests = b.addExecutable(.{
            .name = "tests",
            .target = target,
            .optimize = optimize,
        });

        tests.addIncludePath(deps.doctest.?.path("."));
        tests.linkLibrary(lib);

        tests.addCSourceFiles(.{ .root = og_mlx.path("."), .files = &test_sources, .flags = &CPP_FLAGS });

        if (options.build_metal) {
            tests.addCSourceFile(.{ .file = b.path("tests/metal_tests.cpp"), .flags = &CPP_FLAGS });
        }

        const test_step = b.step("test", "Run library tests");
        const run_cmd = b.addRunArtifact(tests);
        test_step.dependOn(&run_cmd.step);

        b.installArtifact(tests);
    }
}

/////////////////////////////////////////
/// Everything to build metal kernels
///////////////////////////////////////

const MetalKernelBuilder = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    metal_path: []const u8,
    metal_debug: bool,
    metal_jit: bool,
    metal_version: u32,

    fn init(b: *std.Build, target: std.Build.ResolvedTarget, options: BuildOptions) !MetalKernelBuilder {
        const metal_path = b.pathJoin(&.{ b.cache_root.path.?, "metal" });
        return MetalKernelBuilder{
            .b = b,
            .target = target,
            .metal_path = metal_path,
            .metal_debug = options.metal_debug,
            .metal_jit = options.metal_jit,
            .metal_version = try determineMetalVersion(),
        };
    }

    fn determineMetalVersion() !u32 {
        // Run xcrun to determine Metal version
        const result = try std.process.Child.run(.{
            .allocator = std.heap.page_allocator,
            .argv = &[_][]const u8{
                "xcrun", "-sdk", "macosx",   "metal", "-E", "-x", "metal",
                "-P",    "-",    "-include", "metal",
            },
        });
        defer std.heap.page_allocator.free(result.stdout);
        defer std.heap.page_allocator.free(result.stderr);

        // TODO Parse real version from output
        return 310; // Default to Metal 3.1
    }

    fn buildKernel(self: *const MetalKernelBuilder, name: []const u8, deps: []const []const u8) !void {
        var metal_flags = std.ArrayList([]const u8).init(self.b.allocator);
        defer metal_flags.deinit();

        try metal_flags.appendSlice(&[_][]const u8{
            "-Wall",
            "-Wextra",
            "-fno-fast-math",
        });

        if (self.metal_debug) {
            try metal_flags.appendSlice(&[_][]const u8{
                "-gline-tables-only",
                "-frecord-sources",
            });
        }

        // Add version-specific includes
        const version_include = self.getVersionIncludes();
        try metal_flags.appendSlice(&[_][]const u8{
            "-I",
            version_include,
        });

        // Add base headers as dependencies
        var all_deps = std.ArrayList([]const u8).init(self.b.allocator);
        defer all_deps.deinit();
        try all_deps.appendSlice(deps);
        try all_deps.appendSlice(&base_headers);

        const air_path = try std.fmt.allocPrint(
            self.b.allocator,
            "{s}/{s}.air",
            .{ self.metal_path, name },
        );
        defer self.b.allocator.free(air_path);

        // Create metal compilation command
        const metal_cmd = self.b.addSystemCommand(&[_][]const u8{
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
        });

        metal_cmd.addArgs(metal_flags.items);

        metal_cmd.addArg("-c");
        metal_cmd.addFileArg(self.b.path(self.b.fmt("{s}.metal", .{name})));
        metal_cmd.addArg("-o");
        metal_cmd.addFileArg(self.b.path(air_path));

        // Add all dependencies including base headers
        for (all_deps.items) |dep| {
            metal_cmd.addArg("-I");
            metal_cmd.addFileArg(self.b.path(dep));
        }

        // Add STEEL headers to deps when appropriate
        if (std.mem.startsWith(u8, name, "steel/")) {
            try metal_flags.appendSlice(&steel_headers);
        }
    }

    fn buildMetallib(self: *const MetalKernelBuilder) !void {
        const metallib_path = try std.fmt.allocPrint(
            self.b.allocator,
            "{s}/mlx.metallib",
            .{self.metal_path},
        );
        defer self.b.allocator.free(metallib_path);

        // Create metallib command
        const metallib_cmd = self.b.addSystemCommand(&[_][]const u8{
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
        });

        // Add all .air files
        for (air_files) |air| {
            metallib_cmd.addFileArg(self.b.path(air));
        }

        metallib_cmd.addArg("-o");
        metallib_cmd.addFileArg(self.b.path(metallib_path));
    }

    fn buildJitKernels(self: *const MetalKernelBuilder) !void {
        // Add JIT-specific kernels when MLX_METAL_JIT is enabled
        if (self.metal_jit) {
            try self.buildKernel("arange", &[_][]const u8{});
            try self.buildKernel("copy", &[_][]const u8{});
            try self.buildKernel("unary", &[_][]const u8{});
            try self.buildKernel("binary", &[_][]const u8{});
            try self.buildKernel("binary_two", &[_][]const u8{});
            try self.buildKernel("fft", &[_][]const u8{ "kernels/fft/radix.h", "kernels/fft/readwrite.h" });
            try self.buildKernel("ternary", &[_][]const u8{});
            try self.buildKernel("softmax", &[_][]const u8{});
            try self.buildKernel("scan", &[_][]const u8{});
            try self.buildKernel("sort", &[_][]const u8{});
            try self.buildKernel("reduce", &[_][]const u8{
                "kernels/reduction/reduce_all.h",
                "kernels/reduction/reduce_col.h",
                "kernels/reduction/reduce_row.h",
                "kernels/reduction/reduce_init.h",
            });
            try self.buildKernel("steel/gemm/gemm", &[_][]const u8{
                "kernels/steel/utils.h",
                "kernels/steel/gemm/loader.h",
                "kernels/steel/gemm/mma.h",
                "kernels/steel/gemm/params.h",
                "kernels/steel/gemm/transforms.h",
            });
            try self.buildKernel("steel/gemm/kernels/steel_gemm_fused", &[_][]const u8{});
            try self.buildKernel("steel/gemm/kernels/steel_gemm_masked", &[_][]const u8{"kernels/steel/defines.h"});
            try self.buildKernel("steel/gemm/kernels/steel_gemm_splitk", &[_][]const u8{});
            try self.buildKernel("steel/conv/conv", &[_][]const u8{
                "kernels/steel/utils.h",
                "kernels/steel/defines.h",
                "kernels/steel/gemm/mma.h",
                "kernels/steel/gemm/transforms.h",
                "kernels/steel/conv/params.h",
                "kernels/steel/conv/loader.h",
                "kernels/steel/conv/loaders/loader_channel_l.h",
                "kernels/steel/conv/loaders/loader_channel_n.h",
            });
            try self.buildKernel("steel/conv/kernels/steel_conv", &[_][]const u8{});
            try self.buildKernel("steel/conv/kernels/steel_conv_general", &[_][]const u8{
                "kernels/steel/defines.h",
                "kernels/steel/conv/loaders/loader_general.h",
            });
            try self.buildKernel("quantized", &[_][]const u8{});
            try self.buildKernel("gemv_masked", &[_][]const u8{});
        }
    }

    fn buildNonJitKernels(self: *const MetalKernelBuilder) !void {
        // Add these kernels when MLX_METAL_JIT is disabled
        try self.buildKernel("arg_reduce", &[_][]const u8{});
        try self.buildKernel("conv", &[_][]const u8{"steel/conv/params.h"});
        try self.buildKernel("gemv", &[_][]const u8{"steel/utils.h"});
        try self.buildKernel("layer_norm", &[_][]const u8{});
        try self.buildKernel("random", &[_][]const u8{});
        try self.buildKernel("rms_norm", &[_][]const u8{});
        try self.buildKernel("rope", &[_][]const u8{});
        try self.buildKernel("scaled_dot_product_attention", &[_][]const u8{"sdpa_vector.h"});
        try self.buildKernel("steel/attn/kernels/steel_attention", &steel_attn_header);
    }

    fn buildCoreKernels(self: *const MetalKernelBuilder) !void {
        // Build core Metal kernels
        try self.buildKernel("utils", &[_][]const u8{
            "kernels/jit/bf16.h",
            "kernels/metal_3_0/bf16.h",
            "kernels/metal_3_1/bf16.h",
            "kernels/bf16_math.h",
            "kernels/complex.h",
            "kernels/defines.h",
        });
        try self.buildKernel("unary_ops", &[_][]const u8{ "kernels/erf.h", "kernels/expm1f.h" });
        try self.buildKernel("binary_ops", &[_][]const u8{});
        try self.buildKernel("ternary_ops", &[_][]const u8{});
        try self.buildKernel("reduce_utils", &[_][]const u8{ "kernels/atomic.h", "kernels/reduction/ops.h" });
        try self.buildKernel("scatter", &[_][]const u8{"kernels/indexing.h"});
        try self.buildKernel("gather", &[_][]const u8{"kernels/indexing.h"});
        try self.buildKernel("hadamard", &[_][]const u8{});
    }

    fn getVersionIncludes(self: *const MetalKernelBuilder) []const u8 {
        return if (self.metal_version >= 310)
            "upstream/mlx/backend/metal/kernels/metal_3_1"
        else
            "upstream/mlx/backend/metal/kernels/metal_3_0";
    }
};

const steel_headers = [_][]const u8{
    "steel/defines.h",
    "steel/utils.h",
    "steel/conv/conv.h",
    "steel/conv/loader.h",
    "steel/conv/loaders/loader_channel_l.h",
    "steel/conv/loaders/loader_channel_n.h",
    "steel/conv/loaders/loader_general.h",
    "steel/conv/kernels/steel_conv.h",
    "steel/conv/kernels/steel_conv_general.h",
    "steel/gemm/gemm.h",
    "steel/gemm/mma.h",
    "steel/gemm/loader.h",
    "steel/gemm/transforms.h",
    "steel/gemm/kernels/steel_gemm_fused.h",
    "steel/gemm/kernels/steel_gemm_masked.h",
    "steel/gemm/kernels/steel_gemm_splitk.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
};

const steel_attn_header = [_][]const u8{
    "steel/defines.h",
    "steel/utils.h",
    "steel/gemm/gemm.h",
    "steel/gemm/mma.h",
    "steel/gemm/loader.h",
    "steel/gemm/transforms.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/attn/attn.h",
    "steel/attn/loader.h",
    "steel/attn/mma.h",
    "steel/attn/params.h",
    "steel/attn/transforms.h",
    "steel/attn/kernels/steel_attention.h",
};

const base_headers = [_][]const u8{
    "metal_3_1/bf16.h",
    "metal_3_0/bf16.h",
    "bf16_math.h",
    "complex.h",
    "defines.h",
    "expm1f.h",
    "utils.h",
};

///////////////////////////////////////////
/// Build deps like gguf, safetensors etc.
//////////////////////////////////////////

// TODO these are now kinda written as if they have build.zig files, which is not the case
// create a more unified and foolproof way to download and build the correct stuff
const Dependencies = struct {
    fmt: *std.Build.Dependency,
    doctest: ?*std.Build.Dependency,
    // json: ?*std.Build.Dependency,
    gguflib: ?*std.Build.Dependency,
    // metal_cpp: ?*std.Build.Dependency = null,
    // nanobind: ?*std.Build.Dependency = null, this is to build python binding add back later

    fn init(b: *std.Build, options: BuildOptions, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !Dependencies {
        const fmt = b.dependency("fmt", .{
            .target = target,
            .optimize = optimize,
        });

        // TODO fmt moved down, maybe okay to be there
        // const fmt = b.dependency("fmt", .{
        //     .url = "git+https://github.com/fmtlib/fmt.git#10.2.1",
        //     .hash = "1220f6c1f5b8a20f51b2d3534296e2e1b910c3a1f5c8c08f9a8e8523a9c7c5e4d8c8", // You'll need to update this hash
        // });

        const doctest = if (options.build_tests) b.dependency("doctest", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        // TODO figure out later how to include these
        // const json = if (options.build_safetensors) b.dependency("json", .{
        //     .url = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz",
        //     .hash = "...", // You'll need the actual hash
        // }) else null;

        const gguflib = if (options.build_gguf) b.dependency("gguflib", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        // Initialize Metal C++ if needed
        // const metal_cpp = if (options.build_metal) b.dependency("metal_cpp", .{
        //     .url = "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18-beta.zip",
        //     .hash = "...", // Add proper hash
        // }) else null;

        // Initialize nanobind if Python bindings are enabled
        // const nanobind = if (options.build_python_bindings) b.dependency("nanobind", .{
        //     .url = "...", // Add proper URL
        //     .hash = "...", // Add proper hash
        // }) else null;

        return Dependencies{
            .fmt = fmt,
            .doctest = doctest,
            // .json = json,
            .gguflib = gguflib,
            // .metal_cpp = metal_cpp,
            // .nanobind = nanobind,
        };
    }
};

/////////////////////////////////////////
/// Build Preamble
///////////////////////////////////////

fn build_preamble(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, is_darwin: bool) !void {
    const wf = b.addWriteFiles();

    // TODO fix this warning
    // clang++: warning: treating 'c-header' input as 'c++-header' when in C++ mode, this behavior is deprecated [-Wdeprecated]
    const preprocess = b.addSystemCommand(&[_][]const u8{
        "c++",
        "-I",
        b.pathJoin(&.{og_mlx.path(".").getPath(b)}),
        "-E",
        b.pathJoin(&.{ og_mlx.path("mlx").getPath(b), "backend", "common", "compiled_preamble.h" }),
    });

    const std_out_path = preprocess.captureStdOut();

    const read_step = ReadFileStep.create(b, std_out_path);
    read_step.step.dependOn(&preprocess.step);

    const gen_step = GeneratePreambleStep.create(b, read_step, is_darwin, wf, lib);
    gen_step.step.dependOn(&read_step.step);

    // IMPORTANT: Make WriteFile step depend on generate step
    wf.step.dependOn(&gen_step.step);

    const add_step = AddFileStep.create(b, lib, gen_step);
    add_step.step.dependOn(&wf.step);

    lib.step.dependOn(&add_step.step);
}

const ReadFileStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    path: std.Build.LazyPath,
    contents: []const u8 = "",

    fn create(b: *std.Build, path: std.Build.LazyPath) *Self {
        const new = b.allocator.create(Self) catch unreachable;
        new.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "read_file",
                .owner = b,
                .makeFn = make,
            }),
            .path = path,
            .b = b,
        };
        return new;
    }

    fn make(step: *std.Build.Step, prog_node: std.Progress.Node) anyerror!void {
        _ = prog_node;
        const self: *Self = @fieldParentPtr("step", step);
        const path = self.path.getPath(self.b);

        self.contents = try std.fs.cwd().readFileAlloc(
            self.b.allocator,
            path,
            std.math.maxInt(usize),
        );
    }
};

const GeneratePreambleStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    read_step: *ReadFileStep,
    is_darwin: bool,
    wf: *std.Build.Step.WriteFile,
    output_path: std.Build.LazyPath,
    lib: *std.Build.Step.Compile,

    pub fn create(
        b: *std.Build,
        read_step: *ReadFileStep,
        is_darwin: bool,
        wf: *std.Build.Step.WriteFile,
        lib: *std.Build.Step.Compile,
    ) *GeneratePreambleStep {
        const new = b.allocator.create(Self) catch unreachable;

        new.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "generate_preamble",
                .owner = b,
                .makeFn = make,
            }),
            .b = b,
            .read_step = read_step,
            .is_darwin = is_darwin,
            .wf = wf,
            .output_path = undefined, // Will be set in make()
            .lib = lib,
        };

        return new;
    }

    fn make(step: *std.Build.Step, prog_node: std.Progress.Node) anyerror!void {
        _ = prog_node;
        const self: *Self = @fieldParentPtr("step", step);

        var content = std.ArrayList(u8).init(self.b.allocator);
        defer content.deinit();

        try content.appendSlice(
            \\const char* get_kernel_preamble() {
            \\return R"preamble(
            \\
        );

        if (self.is_darwin) {
            try content.appendSlice(
                \\#include <cmath>
                \\#include <complex>
                \\#include <cstdint>
                \\#include <vector>
                \\
            );
        }

        try content.appendSlice(self.read_step.contents);

        try content.appendSlice(
            \\
            \\using namespace mlx::core;
            \\using namespace mlx::core::detail;
            \\)preamble";
            \\}
            \\
        );

        self.output_path = self.wf.add("mlx/backend/common/compiled_preamble.cpp", content.items);
    }
};

const AddFileStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    gen_step: *GeneratePreambleStep,
    lib: *std.Build.Step.Compile,

    fn create(b: *std.Build, lib: *std.Build.Step.Compile, gen_step: *GeneratePreambleStep) *Self {
        const new = b.allocator.create(Self) catch unreachable;
        new.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "add_file",
                .owner = b,
                .makeFn = make,
            }),
            .b = b,
            .lib = lib,
            .gen_step = gen_step,
        };
        return new;
    }

    fn make(step: *std.Build.Step, prog_node: std.Progress.Node) anyerror!void {
        _ = prog_node;
        const self: *Self = @fieldParentPtr("step", step);

        const preamble_cpp_file = self.gen_step.output_path;
        self.lib.addCSourceFile(.{ .file = preamble_cpp_file, .flags = &CPP_FLAGS });
    }
};

////////////////////////////
/// util functions
////////////////////////////

fn checkMacOSSDKVersion() !f32 {
    const result = try std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "xcrun", "-sdk", "macosx", "--show-sdk-version" },
    });
    defer std.heap.page_allocator.free(result.stdout);
    defer std.heap.page_allocator.free(result.stderr);

    const version = try std.fmt.parseFloat(f32, std.mem.trim(u8, result.stdout, " \n\r"));
    if (version < 14.0) {
        @panic("MLX requires macOS SDK >= 14.0 to be built with MLX_BUILD_METAL=ON");
    }
    return version;
}

fn checkMPI() !struct { found: bool, include_path: ?[]const u8 } {
    const mpicxx_result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "mpirun", "--version" },
    }) catch |err| switch (err) {
        error.FileNotFound => return .{ .found = false, .include_path = null },
        else => return err,
    };

    defer std.heap.page_allocator.free(mpicxx_result.stdout);
    defer std.heap.page_allocator.free(mpicxx_result.stderr);

    if (mpicxx_result.term.Exited != 0) {
        return .{ .found = false, .include_path = null };
    }

    // Check if mpirun is OpenMPI
    const mpirun_result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "mpirun", "--version" },
    }) catch |err| switch (err) {
        error.FileNotFound => {
            std.log.warn("MPI found but mpirun is not available. Building without MPI.", .{});
            return .{ .found = false, .include_path = null };
        },
        else => return err,
    };
    defer std.heap.page_allocator.free(mpirun_result.stdout);
    defer std.heap.page_allocator.free(mpirun_result.stderr);

    // Check if output contains "Open MPI"
    if (!std.mem.containsAtLeast(u8, mpirun_result.stdout, 1, "Open MPI")) {
        std.log.warn("MPI which is not OpenMPI found. Building without MPI.", .{});
        return .{ .found = false, .include_path = null };
    }

    // Get MPI include path using mpicxx --showme:incdirs
    const include_result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "mpicxx", "--showme:incdirs" },
    }) catch |err| return err;
    defer std.heap.page_allocator.free(include_result.stdout);
    defer std.heap.page_allocator.free(include_result.stderr);

    const include_path = std.mem.trim(u8, include_result.stdout, " \n\r");

    return .{
        .found = true,
        .include_path = include_path,
    };
}

/////////////////////////
/// Files
/////////////////////////

const core_sources = [_][]const u8{
    "allocator.cpp",
    "array.cpp",
    "compile.cpp",
    "device.cpp",
    "dtype.cpp",
    "einsum.cpp",
    "fast.cpp",
    "fft.cpp",
    "ops.cpp",
    "graph_utils.cpp",
    "primitives.cpp",
    "random.cpp",
    "scheduler.cpp",
    "transforms.cpp",
    "utils.cpp",
    "linalg.cpp",

    // this is technically in the IO dir, but I added it here since it's always required
    "io/load.cpp",
};

// TODO add later
// Add distributed and IO sources
const distributed_sources = [_][]const u8{
    // "distributed/distributed.cpp",
    // Add other distributed source files
};

const metal_sources = [_][]const u8{
    "backend/metal/allocator.cpp",
    "backend/metal/binary.cpp",
    "backend/metal/conv.cpp",
    "backend/metal/compiled.cpp",
    "backend/metal/copy.cpp",
    "backend/metal/custom_kernel.cpp",
    "backend/metal/distributed.cpp",
    "backend/metal/device.cpp",
    "backend/metal/event.cpp",
    "backend/metal/fft.cpp",
    "backend/metal/hadamard.cpp",
    "backend/metal/indexing.cpp",
    "backend/metal/matmul.cpp",
    "backend/metal/scaled_dot_product_attention.cpp",
    "backend/metal/metal.cpp",
    "backend/metal/primitives.cpp",
    "backend/metal/quantized.cpp",
    "backend/metal/normalization.cpp",
    "backend/metal/rope.cpp",
    "backend/metal/scan.cpp",
    "backend/metal/slicing.cpp",
    "backend/metal/softmax.cpp",
    "backend/metal/sort.cpp",
    "backend/metal/reduce.cpp",
    "backend/metal/ternary.cpp",
    "backend/metal/unary.cpp",
    "backend/metal/resident.cpp",
    "backend/metal/utils.cpp",
};

// TODO I probably don't need to have the .air files like this, should be able to grab the list by something else
const air_files = [_][]const u8{
    "arg_reduce.air",
    "conv.air",
    // ... other .air files ...
};

const no_metal_sources = [_][]const u8{
    "backend/no_metal/allocator.cpp",
    "backend/no_metal/event.cpp",
    "backend/no_metal/metal.cpp",
    "backend/no_metal/primitives.cpp",
};

const common_sources = [_][]const u8{
    "backend/common/arg_reduce.cpp",
    "backend/common/binary.cpp",
    "backend/common/compiled.cpp",
    "backend/common/common.cpp",
    "backend/common/conv.cpp",
    "backend/common/copy.cpp",
    "backend/common/eigh.cpp",
    "backend/common/erf.cpp",
    "backend/common/fft.cpp",
    "backend/common/hadamard.cpp",
    "backend/common/masked_mm.cpp",
    "backend/common/primitives.cpp",
    "backend/common/quantized.cpp",
    "backend/common/reduce.cpp",
    "backend/common/reduce_utils.cpp",
    "backend/common/scan.cpp",
    "backend/common/select.cpp",
    "backend/common/slicing.cpp",
    "backend/common/softmax.cpp",
    "backend/common/sort.cpp",
    "backend/common/threefry.cpp",
    "backend/common/indexing.cpp",
    "backend/common/load.cpp",
    "backend/common/qrf.cpp",
    "backend/common/svd.cpp",
    "backend/common/inverse.cpp",
    "backend/common/cholesky.cpp",
    "backend/common/utils.cpp",

    // TODO here should come compiled preamble, but not like this
    // "zig-out/include/mlx/backend/common/compiled_preamble.cpp",
};

const no_cpu_sources = [_][]const u8{
    "backend/no_cpu/primitives.cpp",
    "backend/common/common.cpp",
    "backend/common/compiled.cpp",
    "backend/common/compiled_nocpu.cpp",
    "backend/common/reduce_utils.cpp",
    "backend/common/slicing.cpp",
    "backend/common/utils.cpp",
};

const accelerate_sources = [_][]const u8{
    "backend/accelerate/conv.cpp",
    "backend/accelerate/matmul.cpp",
    "backend/accelerate/primitives.cpp",
    "backend/accelerate/quantized.cpp",
    "backend/accelerate/reduce.cpp",
    "backend/accelerate/softmax.cpp",
};

const test_sources = [_][]const u8{
    "tests/tests.cpp",
    "tests/allocator_tests.cpp",
    "tests/array_tests.cpp",
    "tests/arg_reduce_tests.cpp",
    "tests/autograd_tests.cpp",
    "tests/blas_tests.cpp",
    "tests/compile_tests.cpp",
    "tests/custom_vjp_tests.cpp",
    "tests/creations_tests.cpp",
    "tests/device_tests.cpp",
    "tests/einsum_tests.cpp",
    "tests/eval_tests.cpp",
    "tests/fft_tests.cpp",
    "tests/load_tests.cpp",
    "tests/ops_tests.cpp",
    "tests/random_tests.cpp",
    "tests/scheduler_tests.cpp",
    "tests/utils_tests.cpp",
    "tests/vmap_tests.cpp",
    "tests/linalg_tests.cpp",
};
