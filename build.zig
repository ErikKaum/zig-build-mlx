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
            .build_gguf = b.option(bool, "build-gguf", "Include support for GGUF format") orelse false,
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

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    // TODO: standard optimizer gives a trap exit and null pointer bug, don't know exactly why but I have a hunch
    // const optimize = b.standardOptimizeOption(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;

    // Parse build options using the new fromOptions helper
    const options = BuildOptions.fromOptions(b);

    // Initialize dependencies
    // const deps = try Dependencies.init(b, options);

    // Core MLX library
    const lib = b.addStaticLibrary(.{
        .name = "mlx",
        .target = target,
        .optimize = optimize,
    });

    // TODO this gets all headers, not sure we need all e.g. for metal kernels etc
    lib.installHeadersDirectory(b.path("mlx/mlx"), "mlx", .{});
    lib.addIncludePath(b.path("mlx"));
    lib.linkLibCpp();

    // Adds the fmt headers only
    // TODO figure out best way to download them, or just use a git submodule
    lib.addIncludePath(b.path("fmt/include"));
    // lib.defineCMacro("FMT_HEADER_ONLY", "1");

    // Add core sources
    lib.addCSourceFiles(.{ .files = &core_sources, .flags = &CPP_FLAGS });

    // TODO check that the files included for safetensors and ggufs are really correct
    if (options.build_safetensors) {
        lib.addCSourceFile(.{
            .file = b.path("mlx/mlx/io/safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    } else {
        lib.addCSourceFile(.{
            .file = b.path("mlx/mlx/io/no_safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    }

    if (options.build_gguf) {
        const gguf_sources = [_][]const u8{
            "mlx/mlx/io/gguf.cpp",
            "mlx/mlx/io/gguf_quants.cpp",
        };
        lib.addCSourceFiles(.{ .files = &gguf_sources, .flags = &CPP_FLAGS });
    } else {
        lib.addCSourceFile(.{
            .file = b.path("mlx/mlx/io/no_gguf.cpp"),
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

    try build_preamble(b, lib);

    // should these be done earlier maybe?
    const is_darwin = target.result.isDarwin();
    const is_arm = std.Target.Cpu.Arch.isAARCH64(target.result.cpu.arch);
    const is_x86_64 = std.Target.Cpu.Arch.isX86(target.result.cpu.arch);

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
                .file = b.path("mlx/mlx/backend/metal/jit_kernels.cpp"),
                .flags = &CPP_FLAGS,
            });
        } else {
            lib.addCSourceFile(.{
                .file = b.path("mlx/mlx/backend/metal/nojit_kernels.cpp"),
                .flags = &CPP_FLAGS,
            });
        }
    } else {
        lib.addCSourceFiles(.{ .files = &no_metal_sources, .flags = &CPP_FLAGS });
    }

    if (options.build_cpu) {
        lib.addCSourceFiles(.{ .files = &common_sources, .flags = &CPP_FLAGS });

        // Add IOS-specific sources
        if (target.query.os_tag == std.Target.Os.Tag.ios) { // TODO check if this is way of confirming iOS
            lib.addCSourceFile(.{
                .file = b.path("mlx/mlx/backend/common/compiled_nocpu.cpp"),
                .flags = &CPP_FLAGS,
            });
        } else {
            lib.addCSourceFile(.{
                .file = b.path("mlx/mlx/backend/common/compiled_cpu.cpp"),
                .flags = &CPP_FLAGS,
            });
        }

        // Add default primitives when not using Accelerate
        if (options.build_cpu and !is_darwin) {
            lib.addCSourceFile(.{
                .file = b.path("mlx/mlx/backend/common/default_primitives.cpp"),
                .flags = &CPP_FLAGS,
            });
        }
    } else {
        lib.addCSourceFiles(.{ .files = &no_cpu_sources, .flags = &CPP_FLAGS });
    }

    // Link with system libraries
    if (is_darwin) {
        if (is_arm) {
            lib.linkFramework("Accelerate");
            lib.addCSourceFiles(.{ .files = &accelerate_sources, .flags = &CPP_FLAGS });
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

        tests.addIncludePath(b.path("doctest"));

        tests.addCSourceFiles(.{ .files = &test_sources, .flags = &CPP_FLAGS });

        if (options.build_metal) {
            tests.addCSourceFile(.{ .file = b.path("mlx/tests/metal_tests.cpp"), .flags = &CPP_FLAGS });
        }

        tests.linkLibrary(lib);

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
            "mlx/backend/metal/kernels/metal_3_1"
        else
            "mlx/backend/metal/kernels/metal_3_0";
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
    // fmt: *std.Build.Dependency,
    // doctest: ?*std.Build.Dependency,
    // json: ?*std.Build.Dependency,
    // gguflib: ?*std.Build.Dependency,
    metal_cpp: ?*std.Build.Dependency = null,
    // nanobind: ?*std.Build.Dependency = null, this is to build python binding add back later

    fn init(b: *std.Build, options: BuildOptions) !Dependencies {

        // TODO fmt moved down, maybe okay to be there
        // const fmt = b.dependency("fmt", .{
        //     .url = "git+https://github.com/fmtlib/fmt.git#10.2.1",
        //     .hash = "1220f6c1f5b8a20f51b2d3534296e2e1b910c3a1f5c8c08f9a8e8523a9c7c5e4d8c8", // You'll need to update this hash
        // });

        // I'm pretty sure we don't need it since test are handeled natively in zig
        // Optional dependencies based on build options
        // const doctest = if (options.build_tests) b.dependency("doctest", .{
        //     .url = "https://github.com/onqtam/doctest/archive/ae7a13539fb71f270b87eb2e874fbac80bc8dda2.tar.gz",
        //     .hash = "...", // You'll need the actual hash
        // }) else null;

        // TODO figure out later how to include these
        // const json = if (options.build_safetensors) b.dependency("json", .{
        //     .url = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz",
        //     .hash = "...", // You'll need the actual hash
        // }) else null;

        // TODO same for this one
        // const gguflib = if (options.build_gguf) b.dependency("gguflib", .{
        //     .url = "https://github.com/antirez/gguf-tools/archive/af7d88d808a7608a33723fba067036202910acb3.tar.gz",
        //     .hash = "...", // You'll need the actual hash
        // }) else null;

        // Initialize Metal C++ if needed
        const metal_cpp = if (options.build_metal) b.dependency("metal_cpp", .{
            .url = "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18-beta.zip",
            .hash = "...", // Add proper hash
        }) else null;

        // Initialize nanobind if Python bindings are enabled
        // const nanobind = if (options.build_python_bindings) b.dependency("nanobind", .{
        //     .url = "...", // Add proper URL
        //     .hash = "...", // Add proper hash
        // }) else null;

        return Dependencies{
            // .fmt = fmt,
            // .doctest = doctest,
            // .json = json,
            // .gguflib = gguflib,
            .metal_cpp = metal_cpp,
            // .nanobind = nanobind,
        };
    }
};

/////////////////////////////////////////
/// Build Preamble
///////////////////////////////////////

fn build_preamble(b: *std.Build, lib: *std.Build.Step.Compile) !void {
    const wf = b.addWriteFiles();
    const pkg_path = b.build_root.path.?;
    const is_darwin = true;

    // TODO fix this warning
    // clang++: warning: treating 'c-header' input as 'c++-header' when in C++ mode, this behavior is deprecated [-Wdeprecated]
    const preprocess = b.addSystemCommand(&[_][]const u8{
        "c++",
        "-I",
        b.pathJoin(&.{ pkg_path, "mlx" }),
        "-E",
        b.pathJoin(&.{ pkg_path, "mlx", "mlx", "backend", "common", "compiled_preamble.h" }),
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
    "mlx/mlx/allocator.cpp",
    "mlx/mlx/array.cpp",
    "mlx/mlx/compile.cpp",
    "mlx/mlx/device.cpp",
    "mlx/mlx/dtype.cpp",
    "mlx/mlx/einsum.cpp",
    "mlx/mlx/fast.cpp",
    "mlx/mlx/fft.cpp",
    "mlx/mlx/ops.cpp",
    "mlx/mlx/graph_utils.cpp",
    "mlx/mlx/primitives.cpp",
    "mlx/mlx/random.cpp",
    "mlx/mlx/scheduler.cpp",
    "mlx/mlx/transforms.cpp",
    "mlx/mlx/utils.cpp",
    "mlx/mlx/linalg.cpp",

    // this is technically in the IO dir, but I added it here since it's always required
    "mlx/mlx/io/load.cpp",
};

// TODO add later
// Add distributed and IO sources
const distributed_sources = [_][]const u8{
    // "mlx/mlx/distributed/distributed.cpp",
    // Add other distributed source files
};

const metal_sources = [_][]const u8{
    "mlx/mlx/backend/metal/allocator.cpp",
    "mlx/mlx/backend/metal/binary.cpp",
    "mlx/mlx/backend/metal/compiled.cpp",
    "mlx/mlx/backend/metal/conv.cpp",
    "mlx/mlx/backend/metal/copy.cpp",
    "mlx/mlx/backend/metal/custom_kernel.cpp",
    "mlx/mlx/backend/metal/distributed.cpp",
    "mlx/mlx/backend/metal/device.cpp",
    "mlx/mlx/backend/metal/event.cpp",
    "mlx/mlx/backend/metal/fft.cpp",
    "mlx/mlx/backend/metal/hadamard.cpp",
    "mlx/mlx/backend/metal/indexing.cpp",
    "mlx/mlx/backend/metal/matmul.cpp",
    "mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
    "mlx/mlx/backend/metal/metal.cpp",
    "mlx/mlx/backend/metal/primitives.cpp",
    "mlx/mlx/backend/metal/quantized.cpp",
    "mlx/mlx/backend/metal/normalization.cpp",
    "mlx/mlx/backend/metal/rope.cpp",
    "mlx/mlx/backend/metal/scan.cpp",
    "mlx/mlx/backend/metal/slicing.cpp",
    "mlx/mlx/backend/metal/softmax.cpp",
    "mlx/mlx/backend/metal/sort.cpp",
    "mlx/mlx/backend/metal/reduce.cpp",
    "mlx/mlx/backend/metal/ternary.cpp",
    "mlx/mlx/backend/metal/unary.cpp",
    "mlx/mlx/backend/metal/resident.cpp",
    "mlx/mlx/backend/metal/utils.cpp",
};

// TODO I probably don't need to have the .air files like this, should be able to grab the list by something else
const air_files = [_][]const u8{
    "arg_reduce.air",
    "conv.air",
    // ... other .air files ...
};

const no_metal_sources = [_][]const u8{
    "mlx/mlx/backend/no_metal/allocator.cpp",
    "mlx/mlx/backend/no_metal/event.cpp",
    "mlx/mlx/backend/no_metal/metal.cpp",
    "mlx/mlx/backend/no_metal/primitives.cpp",
};

const common_sources = [_][]const u8{
    "mlx/mlx/backend/common/arg_reduce.cpp",
    "mlx/mlx/backend/common/binary.cpp",
    "mlx/mlx/backend/common/compiled.cpp",
    "mlx/mlx/backend/common/common.cpp",
    "mlx/mlx/backend/common/conv.cpp",
    "mlx/mlx/backend/common/copy.cpp",
    "mlx/mlx/backend/common/eigh.cpp",
    "mlx/mlx/backend/common/erf.cpp",
    "mlx/mlx/backend/common/fft.cpp",
    "mlx/mlx/backend/common/hadamard.cpp",
    "mlx/mlx/backend/common/masked_mm.cpp",
    "mlx/mlx/backend/common/primitives.cpp",
    "mlx/mlx/backend/common/quantized.cpp",
    "mlx/mlx/backend/common/reduce.cpp",
    "mlx/mlx/backend/common/reduce_utils.cpp",
    "mlx/mlx/backend/common/scan.cpp",
    "mlx/mlx/backend/common/select.cpp",
    "mlx/mlx/backend/common/slicing.cpp",
    "mlx/mlx/backend/common/softmax.cpp",
    "mlx/mlx/backend/common/sort.cpp",
    "mlx/mlx/backend/common/threefry.cpp",
    "mlx/mlx/backend/common/indexing.cpp",
    "mlx/mlx/backend/common/load.cpp",
    "mlx/mlx/backend/common/qrf.cpp",
    "mlx/mlx/backend/common/svd.cpp",
    "mlx/mlx/backend/common/inverse.cpp",
    "mlx/mlx/backend/common/cholesky.cpp",
    "mlx/mlx/backend/common/utils.cpp",

    // TODO here should come compiled preamble, but not like this
    // "zig-out/include/mlx/backend/common/compiled_preamble.cpp",
};

const no_cpu_sources = [_][]const u8{
    "mlx/mlx/backend/no_cpu/primitives.cpp",
    "mlx/mlx/backend/common/common.cpp",
    "mlx/mlx/backend/common/compiled.cpp",
    "mlx/mlx/backend/common/compiled_nocpu.cpp",
    "mlx/mlx/backend/common/reduce_utils.cpp",
    "mlx/mlx/backend/common/slicing.cpp",
    "mlx/mlx/backend/common/utils.cpp",
};

const accelerate_sources = [_][]const u8{
    "mlx/mlx/backend/accelerate/conv.cpp",
    "mlx/mlx/backend/accelerate/matmul.cpp",
    "mlx/mlx/backend/accelerate/primitives.cpp",
    "mlx/mlx/backend/accelerate/quantized.cpp",
    "mlx/mlx/backend/accelerate/reduce.cpp",
    "mlx/mlx/backend/accelerate/softmax.cpp",
};

const test_sources = [_][]const u8{
    "mlx/tests/tests.cpp",
    "mlx/tests/allocator_tests.cpp",
    "mlx/tests/array_tests.cpp",
    "mlx/tests/arg_reduce_tests.cpp",
    "mlx/tests/autograd_tests.cpp",
    "mlx/tests/blas_tests.cpp",
    "mlx/tests/compile_tests.cpp",
    "mlx/tests/custom_vjp_tests.cpp",
    "mlx/tests/creations_tests.cpp",
    "mlx/tests/device_tests.cpp",
    "mlx/tests/einsum_tests.cpp",
    "mlx/tests/eval_tests.cpp",
    "mlx/tests/fft_tests.cpp",
    "mlx/tests/load_tests.cpp",
    "mlx/tests/ops_tests.cpp",
    "mlx/tests/random_tests.cpp",
    "mlx/tests/scheduler_tests.cpp",
    "mlx/tests/utils_tests.cpp",
    "mlx/tests/vmap_tests.cpp",
    "mlx/tests/linalg_tests.cpp",
};
