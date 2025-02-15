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
            .build_metal = b.option(bool, "build-metal", "Build metal backend") orelse true,
            .build_cpu = b.option(bool, "build-cpu", "Build cpu backend") orelse true,
            .metal_debug = b.option(bool, "metal-debug", "Enhance metal debug workflow") orelse false,
            .enable_x64_mac = b.option(bool, "enable-x64-mac", "Enable building for x64 macOS") orelse false,
            .build_gguf = b.option(bool, "build-gguf", "Include support for GGUF format") orelse true,
            .build_safetensors = b.option(bool, "build-safetensors", "Include support for safetensors format") orelse true,
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
    lib.defineCMacro("FMT_HEADER_ONLY", "1");

    // TODO this gets all headers, not sure we need all e.g. for metal kernels etc
    lib.installHeadersDirectory(og_mlx.path("."), ".", .{});
    lib.addIncludePath(og_mlx.path("."));
    lib.linkLibCpp();

    // Add core sources
    lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &core_sources, .flags = &CPP_FLAGS });

    if (options.build_safetensors) {
        if (deps.json) |json_dep| {
            lib.addIncludePath(json_dep.path("single_include/nlohmann"));
        }

        lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/safetensors.cpp"),
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
            @panic("MLX requires macOS SDK >= 14.0 to be built with -Dbuild-metal=true");
        }
    }

    // Metal support (Darwin only)
    if (options.build_metal) {
        const root = deps.metal_cpp.?.path(".");
        lib.addIncludePath(root);
        lib.installHeadersDirectory(root, ".", .{ .include_extensions = &.{".hpp"} });

        // TODO have to figure out how the vision OS thing works:
        // /Users/erikkaum/.cache/zig/p/1220d24e8ea45a42f1e5b4928f0991cb2d15fb502e602d57c1551cca4f702398e7f0/mlx/backend/metal/device.cpp:28:56
        lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &metal_sources, .flags = &CPP_FLAGS });

        const metal_lib_path_raw = try runKernelBuild(b, og_mlx, options);
        const metal_lib_path = try std.fmt.allocPrint(b.allocator, "\"{s}\"", .{metal_lib_path_raw});
        lib.defineCMacro("METAL_PATH", metal_lib_path);

        if (options.metal_jit) {
            lib.addCSourceFile(.{ .file = og_mlx.path("mlx/backend/metal/jit_kernels.cpp") });
        } else {
            lib.addCSourceFile(.{ .file = og_mlx.path("mlx/backend/metal/nojit_kernels.cpp") });
        }

        try build_jit_sources(b, lib, og_mlx, options);

        lib.linkFramework("Metal");
        lib.linkFramework("Foundation");
        lib.linkFramework("QuartzCore");
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
            tests.addCSourceFile(.{ .file = og_mlx.path("tests/metal_tests.cpp"), .flags = &CPP_FLAGS });
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

fn runKernelBuild(b: *std.Build, og_mlx: *std.Build.Dependency, options: BuildOptions) ![]const u8 {
    var airFiles = std.ArrayList(std.Build.LazyPath).init(b.allocator);
    defer airFiles.deinit();

    // always build
    try airFiles.append(try buildKernel(b, "arg_reduce", og_mlx));
    try airFiles.append(try buildKernel(b, "conv", og_mlx));
    try airFiles.append(try buildKernel(b, "gemv", og_mlx));
    try airFiles.append(try buildKernel(b, "layer_norm", og_mlx));
    try airFiles.append(try buildKernel(b, "random", og_mlx));
    try airFiles.append(try buildKernel(b, "rms_norm", og_mlx));
    try airFiles.append(try buildKernel(b, "rope", og_mlx));
    try airFiles.append(try buildKernel(b, "scaled_dot_product_attention", og_mlx));
    try airFiles.append(try buildKernel(b, "steel/attn/kernels/steel_attention", og_mlx));

    if (!options.metal_jit) {
        try airFiles.append(try buildKernel(b, "arange", og_mlx));
        try airFiles.append(try buildKernel(b, "binary", og_mlx));
        try airFiles.append(try buildKernel(b, "binary_two", og_mlx));
        try airFiles.append(try buildKernel(b, "copy", og_mlx));
        try airFiles.append(try buildKernel(b, "fft", og_mlx));
        try airFiles.append(try buildKernel(b, "reduce", og_mlx));
        try airFiles.append(try buildKernel(b, "quantized", og_mlx));
        try airFiles.append(try buildKernel(b, "scan", og_mlx));
        try airFiles.append(try buildKernel(b, "softmax", og_mlx));
        try airFiles.append(try buildKernel(b, "sort", og_mlx));
        try airFiles.append(try buildKernel(b, "ternary", og_mlx));
        try airFiles.append(try buildKernel(b, "unary", og_mlx));
        try airFiles.append(try buildKernel(b, "steel/conv/kernels/steel_conv", og_mlx));
        try airFiles.append(try buildKernel(b, "steel/conv/kernels/steel_conv_general", og_mlx));
        try airFiles.append(try buildKernel(b, "steel/gemm/kernels/steel_gemm_fused", og_mlx));
        try airFiles.append(try buildKernel(b, "steel/gemm/kernels/steel_gemm_masked", og_mlx));
        try airFiles.append(try buildKernel(b, "steel/gemm/kernels/steel_gemm_splitk", og_mlx));
        try airFiles.append(try buildKernel(b, "gemv_masked", og_mlx));
    }

    // finally build the metallib which depends on all the air files
    const full_metal_path = try buildMetallib(b, airFiles.items);
    return full_metal_path;
}

// The original MLX uses dependency tracking for file changes, I've chosen to omit this for now
fn buildKernel(b: *std.Build, comptime rel_path: []const u8, og_mlx: *std.Build.Dependency) !std.Build.LazyPath {

    // if name has slashes just grab the last name after the last slash, should be fine
    const name = comptime (if (std.mem.lastIndexOf(u8, rel_path, "/")) |last_slash|
        rel_path[(last_slash + 1)..]
    else
        rel_path);

    var metal_flags = std.ArrayList([]const u8).init(b.allocator);
    defer metal_flags.deinit();

    try metal_flags.appendSlice(&[_][]const u8{
        "-Wall",
        "-Wextra",
        "-fno-fast-math",
    });

    // TODO don't hard code
    const version_include = getVersionIncludes(310);
    try metal_flags.appendSlice(&[_][]const u8{
        "-I",
        og_mlx.path(version_include).getPath(b),
    });

    // In the CMake PROJECT_SOURCE_DIR is always included
    try metal_flags.appendSlice(&[_][]const u8{
        "-I",
        og_mlx.path(".").getPath(b),
    });

    const source_path = "mlx/backend/metal/kernels/" ++ rel_path ++ ".metal";
    const source_path_lazy = og_mlx.path(source_path);

    // Create system command for metal compilation.
    const metal_cmd = b.addSystemCommand(&[_][]const u8{
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
    });
    metal_cmd.addArgs(metal_flags.items);

    metal_cmd.addArg("-c");
    metal_cmd.addArg(source_path_lazy.getPath(b));
    metal_cmd.addArg("-o");
    const out_file_name = name ++ ".air";
    const output_path = metal_cmd.addOutputFileArg(out_file_name);

    // ---------------------------------------------------------------------

    const dest_rel_path = "include/mlx/backend/metal/kernels/" ++ name ++ ".air";
    const metal_install = b.addInstallFile(output_path, dest_rel_path);
    metal_install.step.dependOn(&metal_cmd.step);

    // ---------------------------------------------------------------------

    b.default_step.dependOn(&metal_install.step);

    return output_path;
}

fn buildMetallib(b: *std.Build, air_files: []std.Build.LazyPath) ![]const u8 {
    const metallib_cmd = b.addSystemCommand(&[_][]const u8{
        "xcrun",
        "-sdk",
        "macosx",
        "metallib",
    });

    for (air_files) |air| {
        metallib_cmd.addFileArg(air);
    }

    metallib_cmd.addArg("-o");
    const metallib_file = metallib_cmd.addOutputFileArg("mlx.metallib");

    // Mimic the CMake install rule: install to the lib directory.
    const install = b.addInstallFile(metallib_file, "include/mlx/backend/metal/kernels/mlx.metallib");
    install.step.dependOn(&metallib_cmd.step);
    b.default_step.dependOn(&install.step);

    const full_metal_path = b.getInstallPath(install.dir, install.dest_rel_path);
    return full_metal_path;
}

fn getVersionIncludes(metal_version: u32) []const u8 {
    return if (metal_version >= 310)
        "mlx/backend/metal/kernels/metal_3_1"
    else
        "mlx/backend/metal/kernels/metal_3_0";
}

fn determineMetalVersion(allocator: std.mem.Allocator) u32 {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{
            "zsh",
            "-c",
            "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal -E -x metal -P - | tail -1",
        },
    }) catch |err| {
        std.debug.panic("Failed to get Metal version: {s}", .{@errorName(err)});
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const version_str = std.mem.trim(u8, result.stdout, &std.ascii.whitespace);

    return std.fmt.parseInt(u32, version_str, 10) catch |err| {
        std.debug.panic("Failed to parse Metal version '{s}': {s}", .{ version_str, @errorName(err) });
    };
}

fn build_jit_sources(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, options: BuildOptions) !void {
    try make_jit_source(b, lib, og_mlx, "utils");
    try make_jit_source(b, lib, og_mlx, "unary_ops");
    try make_jit_source(b, lib, og_mlx, "binary_ops");
    try make_jit_source(b, lib, og_mlx, "ternary_ops");
    try make_jit_source(b, lib, og_mlx, "reduce_utils");
    try make_jit_source(b, lib, og_mlx, "scatter");
    try make_jit_source(b, lib, og_mlx, "gather");
    try make_jit_source(b, lib, og_mlx, "hadamard");

    if (options.metal_jit) {
        try make_jit_source(b, lib, og_mlx, "arange");
        try make_jit_source(b, lib, og_mlx, "copy");
        try make_jit_source(b, lib, og_mlx, "unary");
        try make_jit_source(b, lib, og_mlx, "binary");
        try make_jit_source(b, lib, og_mlx, "binary_two");
        try make_jit_source(b, lib, og_mlx, "fft");
        try make_jit_source(b, lib, og_mlx, "ternary");
        try make_jit_source(b, lib, og_mlx, "softmax");
        try make_jit_source(b, lib, og_mlx, "scan");
        try make_jit_source(b, lib, og_mlx, "sort");
        try make_jit_source(b, lib, og_mlx, "reduce");
        try make_jit_source(b, lib, og_mlx, "steel/gemm/gemm");
        try make_jit_source(b, lib, og_mlx, "steel/gemm/kernels/steel_gemm_fused");
        try make_jit_source(b, lib, og_mlx, "steel/gemm/kernels/steel_gemm_masked");
        try make_jit_source(b, lib, og_mlx, "steel/gemm/kernels/steel_gemm_splitk");
        try make_jit_source(b, lib, og_mlx, "steel/conv/kernels/steel_conv");
        try make_jit_source(b, lib, og_mlx, "steel/conv/kernels/steel_conv_general");
        try make_jit_source(b, lib, og_mlx, "quantized");
        try make_jit_source(b, lib, og_mlx, "gemv_masked");
    }
}

fn make_jit_source(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, comptime name: []const u8) !void {
    const wf = b.addWriteFiles();

    const header_file_name = name ++ ".h";
    const source_dir = b.pathJoin(&.{og_mlx.path(".").getPath(b)});
    const jit_includes = og_mlx.path("mlx/backend/metal/kernels/jit").getPath(b);
    const headerPath = b.pathJoin(&.{ og_mlx.path("mlx/backend/metal/kernels").getPath(b), header_file_name });

    // kinda rouge to discard the errros but this is also done in the original MLX repo
    const commandStr = try std.fmt.allocPrint(b.allocator, "{s} -I{s} -I{s} -DMLX_METAL_JIT -E -P {s} 2>/dev/null || true", .{
        // TODO don't hard code this cc compiler
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc",
        source_dir,
        jit_includes,
        headerPath,
    });
    defer b.allocator.free(commandStr);

    const preprocess = b.addSystemCommand(&[_][]const u8{
        "sh",
        "-c",
        commandStr,
    });

    const std_out_path = preprocess.captureStdOut();

    const read_step = ReadFileStep.create(b, std_out_path);
    read_step.step.dependOn(&preprocess.step);

    const gen_step = GenerateMetalJitPreambleStep.create(name, b, read_step, wf, lib);
    gen_step.step.dependOn(&read_step.step);

    // IMPORTANT: Make WriteFile step depend on generate step
    wf.step.dependOn(&gen_step.step);

    const add_step = AddMetalFileStep.create(b, lib, gen_step);
    add_step.step.dependOn(&wf.step);

    lib.step.dependOn(&add_step.step);
}

///////////////////////////////////////////
/// Build deps like gguf, safetensors etc.
//////////////////////////////////////////

const Dependencies = struct {
    fmt: *std.Build.Dependency,
    doctest: ?*std.Build.Dependency,
    json: ?*std.Build.Dependency,
    gguflib: ?*std.Build.Dependency,
    metal_cpp: ?*std.Build.Dependency = null,
    // nanobind: ?*std.Build.Dependency = null, this is to build python binding add back later

    fn init(b: *std.Build, options: BuildOptions, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !Dependencies {
        const fmt = b.dependency("fmt", .{
            .target = target,
            .optimize = optimize,
        });

        const doctest = if (options.build_tests) b.dependency("doctest", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        const json = if (options.build_safetensors) b.dependency("json", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        const gguflib = if (options.build_gguf) b.dependency("gguflib", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        const metal_cpp = if (options.build_metal) b.dependency("metal-cpp", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        // Initialize nanobind if Python bindings are enabled
        // const nanobind = if (options.build_python_bindings) b.dependency("nanobind", .{
        //     .url = "...", // Add proper URL
        //     .hash = "...", // Add proper hash
        // }) else null;

        return Dependencies{
            .fmt = fmt,
            .doctest = doctest,
            .json = json,
            .gguflib = gguflib,
            .metal_cpp = metal_cpp,
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

const GenerateMetalJitPreambleStep = struct {
    const Self = @This();

    name: []const u8,
    step: std.Build.Step,
    b: *std.Build,
    read_step: *ReadFileStep,
    wf: *std.Build.Step.WriteFile,
    output_path: std.Build.LazyPath,
    lib: *std.Build.Step.Compile,

    pub fn create(
        name: []const u8,
        b: *std.Build,
        read_step: *ReadFileStep,
        wf: *std.Build.Step.WriteFile,
        lib: *std.Build.Step.Compile,
    ) *GenerateMetalJitPreambleStep {
        const new = b.allocator.create(Self) catch unreachable;

        new.* = .{
            .name = name,
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "generate_metal_jit_preamble",
                .owner = b,
                .makeFn = make,
            }),
            .b = b,
            .read_step = read_step,
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
            \\namespace mlx::core::metal {
            \\
            \\
        );

        const line = try std.fmt.allocPrint(
            self.b.allocator,
            "const char* {s} () {{\nreturn R\"preamble(\n",
            .{self.name},
        );

        try content.appendSlice(line);

        try content.appendSlice(self.read_step.contents);

        try content.appendSlice(
            \\)preamble";
            \\}
            \\
            \\} // namespace mlx::core::metal
            \\
        );

        // The output path is now in the .zig_cache, which may or may not be okay
        const output_path = try std.fmt.allocPrint(self.b.allocator, "mlx/backend/metal/jit/{s}.cpp", .{self.name});
        self.output_path = self.wf.add(output_path, content.items);
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

        // The output path is now in the .zig_cache, which may or may not be okay
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

const AddMetalFileStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    gen_step: *GenerateMetalJitPreambleStep,
    lib: *std.Build.Step.Compile,

    fn create(b: *std.Build, lib: *std.Build.Step.Compile, gen_step: *GenerateMetalJitPreambleStep) *Self {
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
