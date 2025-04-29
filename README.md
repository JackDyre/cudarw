# CudaRw

CudaRw is a dual read-write lock backed by a host buffer and a gpu buffer that lazily keeps them in sync

CudaRw depends on [cudaz](https://github.com/akhildevelops/cudaz) to provide CUDA functionality

## Building

Build CudaRw with `zig build`

## Depending

Add CudaRw as a dependency with `zig fetch --save git+https://github.com/JackDyre/cudarw`

Add it to your `build.zig`:
```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "proj",
        .root_module = exe_mod,
    });

    const cudarw_dep = b.dependency("cudarw", .{ .optimize = optimize, .target = target });
    const cudarw_mod = cudarw_dep.module("cudarw");

    exe_mod.addImport("cudarw", cudarw_mod);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
```

## Example

```zig
const std = @import("std");
const crw = @import("cudarw");
const Cuda = crw.Cuda;
const CudaRw = crw.CudaRw;

pub fn main() !void {
    const device: Cuda.Device = try Cuda.Device.default();
    defer device.deinit();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var cudarw = CudaRw(usize).init(allocator, device, 16);
    defer cudarw.deinit();

    {
        var guard = try cudarw.hostWriteLock();
        defer guard.release();

        const buf = guard.get();

        for (buf) |*num| {
            num.* = 5;
        }
    }

    {
        var guard = try cudarw.hostReadLock();
        defer guard.release();

        const buf = guard.get();
        std.debug.print("{any}\n", .{buf});
    }
}
```
