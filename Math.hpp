#pragma once

struct Vec3 {
    float x, y, z;

    [[nodiscard]] Vec3 operator+(const Vec3 other) const noexcept { return Vec3{x + other.x, y + other.y, z + other.z}; }
    [[nodiscard]] Vec3 operator+(const float s) const noexcept { return Vec3{x + s, y + s, z + s}; }
    [[nodiscard]] Vec3 &operator+=(const Vec3 other) noexcept { return x += other.x, y += other.y, z += other.z, *this; }
    [[nodiscard]] Vec3 operator*(const float s) const noexcept { return Vec3{x * s, y * s, z * s}; }
    [[nodiscard]] float dot(const Vec3 other) const noexcept { return x * other.x + y * other.y + z * other.z; }
    [[nodiscard]] Vec3 cross(const Vec3 other) const noexcept { return Vec3{y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x}; }
};

struct Quat {
    float x, y, z, w;

    [[nodiscard]] static Quat identity() noexcept { return Quat{0.f, 0.f, 0.f, 1.f}; }
    [[nodiscard]] Quat operator*(const Quat &other) const noexcept { return Quat{
        w * other.x + x * other.w + y * other.z - z * other.y,
        w * other.y - x * other.z + y * other.w + z * other.x,
        w * other.z + x * other.y - y * other.x + z * other.w,
        w * other.w - x * other.x - y * other.y - z * other.z
    }; }
    [[nodiscard]] Vec3 rotate(const Vec3 &v) const noexcept {
        Vec3 tmp{x, y, z};
        return v + tmp.cross(tmp.cross(v) + v * w) * 2.f;
    };
};

struct BoundaryBox {
    Vec3 min, max;

    [[nodiscard]] bool contains(const Vec3 &p) const noexcept {
        return p.x >= min.x && p.x <= max.x &&
               p.y >= min.y && p.y <= max.y &&
               p.z >= min.z && p.z <= max.z;
    }
};
