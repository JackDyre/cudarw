const std = @import("std");

pub const Cuda = @import("cudaz");

pub fn CudaRw(comptime T: type) type {
    return struct {
        lock: std.Thread.RwLock = std.Thread.RwLock{},
        alloc_lock: std.Thread.Mutex = std.Thread.Mutex{},
        len: usize,

        host_slice: ?[]T = null,
        device_slice: ?Cuda.Cudaslice(T) = null,

        allocator: std.mem.Allocator,
        device: Cuda.Device,

        host_stale: bool = true,
        device_stale: bool = true,

        pub fn init(allocator: std.mem.Allocator, device: Cuda.Device, len: usize) Self {
            return Self{
                .len = len,
                .allocator = allocator,
                .device = device,
            };
        }

        pub fn deinit(self: *Self) void {
            self.lock.lock();

            if (self.host_slice) |slice| self.allocator.free(slice);
            if (self.device_slice) |slice| Cuda.Device.free(slice.device_ptr) catch @panic("unable to free memory");

            self.lock.unlock();
        }

        fn getHostSlice(self: *Self, allow_uninit: bool) ![]T {
            self.alloc_lock.lock();
            defer self.alloc_lock.unlock();

            const doCopy = self.host_stale and !self.device_stale;

            if (self.host_slice == null) {
                if (!allow_uninit) {
                    return error.ReadLockUnitMemory;
                }
                self.host_slice = try self.allocator.alloc(T, self.len);
            }

            if (doCopy) {
                try Cuda.Device.dtohCopyInto(T, self.device_slice.?, self.host_slice.?);
            }

            self.host_stale = false;
            return self.host_slice.?;
        }

        fn getDeviceSlice(self: *Self, allow_uninit: bool) !Cuda.Cudaslice(T) {
            self.alloc_lock.lock();
            defer self.alloc_lock.unlock();

            const doCopy = self.device_stale and !self.host_stale;

            if (self.device_slice == null) {
                if (!allow_uninit) {
                    return error.ReadLockUnitMemory;
                }
                self.device_slice = try self.device.alloc(T, self.len);
            }

            if (doCopy) {
                try Cuda.Device.htodCopyInto(T, self.host_slice.?, self.device_slice.?);
            }

            self.device_stale = false;
            return self.device_slice.?;
        }

        pub fn hostReadLock(self: *Self) !HostReadGuard {
            self.lock.lockShared();
            return HostReadGuard.new(self) catch |err| {
                self.lock.unlockShared();
                return err;
            };
        }

        pub fn tryHostReadLock(self: *Self) !?HostReadGuard {
            if (!self.lock.tryLockShared()) return null;
            return HostReadGuard.new(self) catch |err| {
                self.lock.unlockShared();
                return err;
            };
        }

        pub fn hostWriteLock(self: *Self) !HostWriteGuard {
            self.lock.lock();
            return HostWriteGuard.new(self) catch |err| {
                self.lock.unlock();
                return err;
            };
        }

        pub fn tryHostWriteLock(self: *Self) !?HostWriteGuard {
            if (!self.lock.tryLock()) return null;
            return HostWriteGuard.new(self) catch |err| {
                self.lock.unlock();
                return err;
            };
        }

        pub fn deviceReadLock(self: *Self) !DeviceReadGuard {
            self.lock.lockShared();
            return DeviceReadGuard.new(self) catch |err| {
                self.lock.unlockShared();
                return err;
            };
        }

        pub fn tryDeviceReadLock(self: *Self) !?DeviceReadGuard {
            if (!self.lock.tryLockShared()) return null;
            return DeviceReadGuard.new(self) catch |err| {
                self.lock.unlockShared();
                return err;
            };
        }

        pub fn deviceWriteLock(self: *Self) !DeviceWriteGuard {
            self.lock.lock();
            return DeviceWriteGuard.new(self) catch |err| {
                self.lock.unlock();
                return err;
            };
        }

        pub fn tryDeviceWriteLock(self: *Self) !?DeviceWriteGuard {
            if (!self.lock.tryLock()) return null;
            return DeviceWriteGuard.new(self) catch |err| {
                self.lock.unlock();
                return err;
            };
        }

        pub const HostReadGuard = struct {
            slice: []const T,
            lock: *Self,

            fn new(self: *Self) !HostReadGuard {
                return HostReadGuard{
                    .slice = try self.getHostSlice(false),
                    .lock = self,
                };
            }
            pub fn get(guard: *HostReadGuard) []const T {
                return guard.slice;
            }
            pub fn release(guard: *HostReadGuard) void {
                guard.lock.lock.unlockShared();
            }
        };

        pub const HostWriteGuard = struct {
            slice: []T,
            lock: *Self,

            fn new(self: *Self) !HostWriteGuard {
                return HostWriteGuard{
                    .slice = try self.getHostSlice(true),
                    .lock = self,
                };
            }
            pub fn get(guard: *HostWriteGuard) []T {
                return guard.slice;
            }
            pub fn release(guard: *HostWriteGuard) void {
                guard.lock.device_stale = true;
                guard.lock.lock.unlock();
            }
        };

        pub const DeviceReadGuard = struct {
            slice: Cuda.Cudaslice(T),
            lock: *Self,

            fn new(self: *Self) !DeviceReadGuard {
                return DeviceReadGuard{
                    .slice = try self.getDeviceSlice(false),
                    .lock = self,
                };
            }
            pub fn get(guard: *DeviceReadGuard) Cuda.Cudaslice(T) {
                return guard.slice;
            }
            pub fn release(guard: *DeviceReadGuard) void {
                guard.lock.lock.unlockShared();
            }
        };

        pub const DeviceWriteGuard = struct {
            slice: Cuda.Cudaslice(T),
            lock: *Self,

            fn new(self: *Self) !DeviceWriteGuard {
                return DeviceWriteGuard{
                    .slice = try self.getDeviceSlice(true),
                    .lock = self,
                };
            }
            pub fn get(guard: *DeviceWriteGuard) Cuda.Cudaslice(T) {
                return guard.slice;
            }
            pub fn release(guard: *DeviceWriteGuard) void {
                guard.lock.host_stale = true;
                guard.lock.lock.unlock();
            }
        };

        const Self = @This();
    };
}
