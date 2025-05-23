const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.addModule("cudarw", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const cudaz_dep = b.dependency("cudaz", .{ .optimize = optimize, .target = target });
    const cudaz_module = cudaz_dep.module("cudaz");

    lib_mod.addImport("cudaz", cudaz_module);

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "cudarw",
        .root_module = lib_mod,
    });

    b.installArtifact(lib);

    lib.linkLibC();
    lib.linkSystemLibrary("cuda");
    lib.linkSystemLibrary("nvrtc");

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
