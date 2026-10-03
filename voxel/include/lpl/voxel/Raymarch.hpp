/**
 * @file Raymarch.hpp
 * @brief Walking a ray through the samples and accumulating what it passes through.
 *
 * @warning **There is no mesh, and that is a decision the data forced.** An isosurface has to pick
 * a density and call everything above it solid; on a real carbonised scroll the sheets are bright
 * ribbons floating in a darker medium that is not empty, so every cut either fuses the ribbons
 * into one blob or shreds them into fragments -- measured, on one slice, at every threshold from
 * 130 to 198. Direct volume rendering never makes that choice: it shows the ribbon, its soft edge,
 * and the fact that the edge is soft. For an instrument somebody will draw conclusions from, a
 * surface that asserts a boundary the scan never resolved is worse than no surface at all.
 *
 * @warning **The cost is pixels, not triangles, and that is why it fits.** The same field meshed
 * at full resolution emits about a million faces per brick -- measured, not estimated -- which no
 * software rasteriser and no immediate-mode context will draw. A ray costs what the screen costs,
 * and the screen is small.
 *
 * @warning **This is float, and float is allowed here because none of it is authoritative.**
 * Nothing a ray computes flows back into simulation state. The engine's fixed-point contract
 * covers what decides the world, not what looks at it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_RAYMARCH_HPP
#    define LPL_VOXEL_RAYMARCH_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/voxel/Mosaic.hpp>
#    include <lpl/voxel/Transfer.hpp>
#    include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

/**
 * @struct Eye
 * @brief Where the camera is and what it is looking at, in walked metres.
 */
struct Eye final {
    math::Vec3<core::f32> position{}; ///< Metres.
    math::Vec3<core::f32> forward{0.0f, 0.0f, 1.0f};
    math::Vec3<core::f32> right{1.0f, 0.0f, 0.0f};
    math::Vec3<core::f32> up{0.0f, 1.0f, 0.0f};
    core::f32 horizontalFieldOfView{1.0472f}; ///< Radians; 60 degrees.
};

/**
 * @struct ImagePlane
 * @brief How the pixels of one frame map to directions from an @ref Eye, both ways.
 *
 * The ray through a point of the frame is `forward + right * rightSlope + up * upSlope`, unnormalised.
 * The marcher casts its rays through it and a surface is projected with it, so a triangle lands on
 * the pixels whose rays reach it; two derivations of the same projection drift apart by a pixel, and
 * nobody can say why the surface sits beside the scan.
 *
 * Positions on the frame are in pixels from its top-left corner, so the centre of pixel (px, py) is
 * (px + 0.5, py + 0.5).
 *
 * @pre Made by @ref imagePlane: a default one divides by zero.
 */
struct ImagePlane final {
    core::f32 tanHalfFieldOfView{0.0f};
    core::f32 aspect{0.0f}; ///< Height over width.
    core::f32 width{0.0f};
    core::f32 height{0.0f};

    /// @return Offset along @ref Eye::right per unit along @ref Eye::forward, at frame column @p x.
    [[nodiscard]] constexpr core::f32 rightSlope(core::f32 x) const noexcept
    {
        return (2.0f * x / width - 1.0f) * tanHalfFieldOfView;
    }

    /// @return Offset along @ref Eye::up per unit along @ref Eye::forward, at frame row @p y.
    [[nodiscard]] constexpr core::f32 upSlope(core::f32 y) const noexcept
    {
        return (1.0f - 2.0f * y / height) * aspect * tanHalfFieldOfView;
    }

    /// @return The frame column whose @ref rightSlope is @p slope.
    [[nodiscard]] constexpr core::f32 column(core::f32 slope) const noexcept
    {
        return (slope / tanHalfFieldOfView + 1.0f) * 0.5f * width;
    }

    /// @return The frame row whose @ref upSlope is @p slope.
    [[nodiscard]] constexpr core::f32 row(core::f32 slope) const noexcept
    {
        return (1.0f - slope / (aspect * tanHalfFieldOfView)) * 0.5f * height;
    }
};

/**
 * @brief The image plane of a @p width by @p height frame seen from @p eye.
 *
 * @pre @p width and @p height are not zero, and the field of view is in (0, pi).
 */
[[nodiscard]] ImagePlane imagePlane(const Eye &eye, core::u32 width, core::u32 height) noexcept;

/**
 * @class FreeCamera
 * @brief A position and two angles, turned into an @ref Eye.
 *
 * @warning **The angles are wrapped before any polynomial touches them, and that is not a
 * formality.** A camera that keeps turning accumulates yaw without bound, and every cheap sine is
 * only good near zero. This engine has already shipped that exact bug once: its CORDIC did not
 * reduce its argument, so past about a hundred degrees the sine and cosine FROZE and a body kept
 * walking in the direction it had at a hundred degrees. It reads as dead controls, not as bad
 * arithmetic, which is why it survived -- the first sixty degrees of every turn work perfectly.
 * That is also why the angles are private: only @ref turn keeps them wrapped and clamped.
 *
 * @warning Pitch is CLAMPED rather than wrapped, because a camera that rolls over the vertical
 * flips its horizon and there is no reading of the controls that recovers from it.
 *
 * @warning No collision, deliberately. A volume is a scan, not a place with floors, and an
 * instrument whose operator can be stopped by a wall is a worse instrument. Flight is the mode,
 * not a cheat.
 */
class FreeCamera final {
public:
    math::Vec3<core::f32> position{};
    core::f32 horizontalFieldOfView{1.0472f};

    /// @brief Turns by @p dYaw and @p dPitch, wrapping the first and clamping the second.
    void turn(core::f32 dYaw, core::f32 dPitch) noexcept;

    /// @brief Moves along the camera's own axes, plus the world vertical.
    void move(core::f32 forward, core::f32 strafe, core::f32 rise, core::f32 distance) noexcept;

    [[nodiscard]] Eye eye() const noexcept;

    /// @return Radians about the world vertical, as @ref wrapAngle leaves it.
    [[nodiscard]] core::f32 yaw() const noexcept { return _yaw; }

    /// @return Radians, clamped just short of straight up and straight down.
    [[nodiscard]] core::f32 pitch() const noexcept { return _pitch; }

private:
    core::f32 _yaw{0.0f};
    core::f32 _pitch{0.0f};
};

/**
 * @brief The same angle, in [-pi, pi] once rounded to float.
 *
 * @return 0 for an angle that is not finite or beyond 1e8 turns (about 6e8 radians), well past
 *         the point where a float carries any fraction of a turn: a direction derived from it then
 *         stays defined instead of freezing or going NaN.
 */
[[nodiscard]] core::f32 wrapAngle(core::f32 radians) noexcept;

/**
 * @brief Sine of any angle, without libm, correct over the whole circle.
 *
 * Exposed because the camera is not the only thing that needs it and a second copy would be a
 * second chance to leave out the reduction.
 *
 * @return The sine of @ref wrapAngle(@p radians), to about two parts in ten thousand.
 */
[[nodiscard]] core::f32 wrappedSine(core::f32 radians) noexcept;

/// @brief Cosine of any angle, without libm, with the accuracy of @ref wrappedSine.
[[nodiscard]] core::f32 wrappedCosine(core::f32 radians) noexcept;

/**
 * @enum DebugView
 * @brief What to paint instead of the picture, when the picture is the thing under suspicion.
 *
 * @warning **This exists because reading the code produced four wrong diagnoses in a row.** The
 * first real renders came out in rectangular patches; the level of detail was blamed twice, the
 * brick-skip jump once, and a false gradient at unresident neighbours once. Each was a real defect
 * and none was the cause. A view that paints the intermediate quantity settles in one frame what a
 * day of reasoning did not: the shading term alone came out as large grey rectangles, which is
 * what finally pointed at the gradient stencil.
 */
enum class DebugView : core::u8 {
    Off = 0,           ///< The picture.
    Level,             ///< Which pyramid level answered, as a grey ramp over every possible level.
    GradientMagnitude, ///< How strong the local gradient is at the first sample that paints.
    Normal,            ///< The surface normal as colour: x, y, z into red, green, blue.
    Shade,             ///< The lighting term alone.
    StepCount,         ///< How many samples the ray took.
};

/// Traced surfaces one frame can composite. A comparison needs two; the other two are headroom.
inline constexpr core::u32 kMaxSurfaceLayers = 4u;

/**
 * @struct SurfaceTexture
 * @brief Where each pixel of a rasterised surface sits in that surface's own flattening.
 */
struct SurfaceTexture final {
    const core::f32 *u{nullptr};
    const core::f32 *v{nullptr};
};

/**
 * @struct SurfaceLayer
 * @brief One traced surface, already rasterised into per-pixel depth.
 *
 * @warning **A traced surface is an INTRUSION and must read as one.** Whether it was walked by
 * this tool or loaded from somebody's file, it is an inference about where a sheet goes, and a
 * reader has to tell it from the samples at a glance -- which is why each layer carries its own
 * colour, and why @ref opacity is allowed to be well under one. A surface you can see through is
 * a surface you can check against what is behind it; an opaque one is a claim that hides its own
 * evidence.
 */
struct SurfaceLayer final {
    const core::f32 *depth{nullptr};  ///< Metres along the ray; a very large value means "none here".
    const core::f32 *facing{nullptr}; ///< |N . V| for shading. Optional.
    core::f32 red{0.35f};
    core::f32 green{0.85f};
    core::f32 blue{1.00f};
    core::f32 opacity{0.55f};

    /**
     * Draw through whatever is in front.
     *
     * @warning **This deliberately breaks the property that makes the surface checkable**, so it
     * is off by default and named for what it does. Composited honestly, a traced sheet is
     * occluded by the matter between it and the eye -- correct, and inside a dense scroll it often
     * means you cannot see it at all. Wanting to see it anyway is legitimate; making it the
     * default is not, because a picture where an inference is never hidden by the evidence is a
     * picture that cannot disagree with it.
     */
    bool throughMatter{false};

    /**
     * A prediction painted ON this surface, in its own flattened coordinates.
     *
     * @warning **This is where the corpus's ink maps actually live, and it is why they could not
     * simply be poured into the volume.** A prediction is produced on a flattened segment, not in
     * three dimensions -- lifting it back needs the segment's own flattening, which is the texture
     * coordinates the mesh carries. Painting it here, through those coordinates, is the only route
     * that does not invent a registration.
     *
     * @warning Everything the volume overlay promises holds here too: null by default, and
     * @ref inkConfidence multiplies the tint rather than sitting in a caption.
     * @see MarchParams::overlay
     */
    const core::u8 *ink{nullptr};
    core::u32 inkWidth{0};
    core::u32 inkHeight{0};

    /// What the map is worth, in [0,1], from a measurement made elsewhere. Zero paints nothing.
    core::f32 inkConfidence{0.0f};

    core::u8 inkFloor{128u}; ///< Below this the map is saying nothing, and nothing is painted.
    core::f32 inkRed{0.05f};
    core::f32 inkGreen{0.02f};
    core::f32 inkBlue{0.02f};

    [[nodiscard]] constexpr bool valid() const noexcept { return depth != nullptr; }

    [[nodiscard]] constexpr bool textured() const noexcept
    {
        return ink != nullptr && inkWidth != 0u && inkHeight != 0u && inkConfidence > 0.0f;
    }
};

/**
 * @struct MarchParams
 * @brief How finely to walk, how soon to stop, and how far to look.
 */
struct MarchParams final {
    core::f32 stepSamples{1.0f}; ///< Step length in level-0 samples. Below 1 costs and shows nothing new.
    core::f32 opaqueAt{0.995f};  ///< Accumulated alpha at which a ray stops. Early termination.
    core::f32 maxDistanceMetres{4000.0f};
    core::u32 maxSteps{4096u};         ///< Hard stop, so a bad ray cannot hang a frame.
    core::u32 background{0xFF06070Au}; ///< 0xAARRGGBB painted where nothing was hit.
    bool trilinear{true};              ///< Off gives nearest-sample: faster, and visibly cubic.

    /**
     * How much of the colour comes from shading the local gradient, in [0,1].
     *
     * @warning **Without this a real scan renders as fog, and the fog is convincing.** Pure
     * emission and absorption over a field whose density varies by a few percent produces a smooth
     * wash: every structure is there, in the numbers, and none of it is legible. The gradient of
     * the field is the normal of the surface passing through the sample, and lighting it is what
     * turns a wash into sheets. This was measured on a real scroll -- the first frames out of this
     * renderer were a plausible cream-coloured haze.
     */
    core::f32 shading{0.85f};

    /**
     * How much the gradient magnitude raises opacity, in [0,1].
     *
     * Classic gradient-magnitude modulation: homogeneous matter stays translucent and boundaries
     * turn solid, so a ray reaches the first real surface instead of drowning in the medium
     * before it gets there.
     */
    core::f32 boundaryOpacity{0.7f};

    /**
     * Light kept at the eye rather than in the world.
     *
     * @warning There is no sun inside a scanned object, and inventing one would put a shadow
     * direction into a picture somebody is going to draw conclusions from. A headlamp states what
     * it is: the operator's own light, so what is bright is what faces them.
     */
    core::f32 ambient{0.25f};

    /**
     * Half-width of the gradient stencil, in level-0 samples.
     *
     * @warning **One is too small on real data, and the failure looks like faceting rather than
     * like noise.** A byte sample with low local contrast gives a nearest-neighbour difference of
     * 0, 1 or 2 over one sample, so the normal lands on a handful of directions and the picture
     * comes out in flat polygons. Two lifts the difference, taken over four samples, clear of
     * that floor without smoothing away the sheet it is supposed to be lighting -- and it is what
     * lets the six probes stay NEAREST samples: interpolating them made each one eight scattered
     * reads, in the costliest part of a frame (@see shadingCutoff).
     */
    core::f32 gradientSpread{2.0f};

    /**
     * Weighted contribution below which a sample is composited without shading it.
     *
     * @warning **This is the frame's biggest lever, and it is a quality knob dressed as a speed
     * one.** Measured on the bench: the gradient is 73 % of a frame while only twelve per cent of
     * samples reach it. A sample whose weighted contribution is a thousandth of a channel cannot
     * change the pixel, and shading it costs six scattered reads. Raising this trades shading
     * detail deep inside an already-opaque ray for time; zero shades everything.
     */
    core::f32 shadingCutoff{0.0015f};

    /**
     * Leave an occupancy cell in one comparison when nothing in it can paint.
     * @see BrickView::occupancy for why the cells, and not the bricks, carry the medium.
     *
     * @warning **Exact for nearest samples, not for trilinear ones.** A cell is skipped only when
     * its HIGHEST sample cannot reach the visible band, and with nearest sampling the frame is
     * bit-identical with or without the skip. A trilinear point near a cell's face interpolates
     * toward the next cell, so a skipped cell can still hold points that would have painted a
     * little: measured at **at most ten levels out of 255** against a 3.4x cut in samples. It is
     * not the step phase, as was once thought: every hop lands on the step grid. On by default
     * because the trade is good and the difference is small; switchable because a tool whose job
     * is a reproducible measurement should be able to decline it.
     */
    bool skipEmptyCells{true};

    /**
     * Width of the band, in level-0 samples, over which a fine brick fades into the coarse one
     * behind it. Zero switches levels hard.
     *
     * @warning **Without this a level of detail is a visible cube.** The fine ring around the eye
     * is a box, and where it stops the density jumps to whatever the coarser brick averaged --
     * which paints a rectangular brightness step in the middle of a scan and reads as structure
     * that is not there. It was the most obvious defect in the first real renders. The band only
     * applies where the detail actually STOPS: if the neighbour at the same level is resident
     * there is no seam, and softening across it would throw away detail that is present.
     */
    core::f32 levelBlendSamples{24.0f};

    /// @see DebugView. Off in every normal frame.
    DebugView debug{DebugView::Off};

    /**
     * A second field, co-registered with the first, painted on top of what it agrees with.
     *
     * @warning **An overlay is somebody's INFERENCE about the subject, and the renderer must never
     * let it look like the subject.** A prediction volume is produced by a model, and a model can
     * be confidently wrong in a way no inspection of its output reveals -- established in this
     * corpus by a negative witness, where a model asked about a surface with no writing anywhere
     * near it produced a different, convincing structure every time. So: this is null by default,
     * a frame with it null is the plain scan and that is the mode a reader gets unless they ask
     * for otherwise, and @ref overlayConfidence is applied to the tint rather than hidden in a
     * caption. A viewer that painted a prediction as fact would be an instrument that lies.
     */
    const BrickMosaic *overlay{nullptr};

    core::f32 overlayRed{0.95f}; ///< Tint applied where the overlay is strong.
    core::f32 overlayGreen{0.25f};
    core::f32 overlayBlue{0.20f};

    /**
     * How far the tint is allowed to go, in [0,1].
     *
     * @warning Set it from what is actually known about the map being shown -- its measured area
     * under the curve, say -- and not to taste. A prediction of 0.55 painted as solidly as one of
     * 0.95 has had its uncertainty removed by the renderer.
     */
    core::f32 overlayConfidence{0.0f};

    /// Overlay value at and above which the tint is at full strength.
    core::u8 overlayFull{200u};

    /// Overlay value below which nothing is tinted at all.
    core::u8 overlayFloor{128u};

    /**
     * Traced surfaces to composite, at most @ref kMaxSurfaceLayers of them.
     *
     * @warning **Two at once is the point, not an extra.** The question a researcher has is
     * whether their trace and somebody else's follow the same sheet, and two pictures of two
     * surfaces answer it badly. Side by side in the same volume, in different colours, it is
     * answered by looking -- and @ref compareSurfaces puts a number on what the eye sees.
     */
    SurfaceLayer surfaces[kMaxSurfaceLayers]{};
    core::u32 surfaceCount{0};

    /**
     * Per-layer texture coordinates from the rasteriser, or null when no layer carries a
     * prediction. Parallel to @ref surfaces.
     */
    const SurfaceTexture *surfaceTexture{nullptr};
};

/**
 * @struct MarchReport
 * @brief What a frame actually did, for a readout that reports rather than asserts.
 *
 * @warning These are what make an image believable. A frame that never entered the volume produces
 * a perfectly stable picture and a perfectly stable signature, so a test that folds only pixels
 * passes against a camera pointing at nothing.
 */
struct MarchReport final {
    core::u64 rays{0};          ///< Pixels traced.
    core::u64 steps{0};         ///< Samples taken.
    core::u64 skippedBricks{0}; ///< Bricks jumped because nothing in them is visible.
    core::u64 skippedCells{0};  ///< Occupancy cells jumped. The medium, which is most of a scan.
    core::u64 missingBricks{0}; ///< Bricks a ray wanted and the mosaic did not have.
    core::u64 saturated{0};     ///< Rays that stopped early on opacity.
    core::u64 escaped{0};       ///< Rays that left the volume still transparent.
    core::u64 gradients{0};     ///< Samples whose gradient was computed; the expensive ones.
    core::u64 overlaid{0};      ///< Samples the overlay tinted. Zero means nothing was shown.
    core::u64 surfaceHits{0};   ///< Rays that crossed a traced surface. Zero means none was in view.
    core::u64 inkPainted{0};    ///< Pixels the surface prediction tinted. Zero means it showed nothing.
};

/**
 * @brief Renders the resident mosaic into @p pixels.
 *
 * @param mosaic    Resident bricks. Nothing is fetched; a brick that is not here is a hole the ray
 *                  passes through, counted in @ref MarchReport::missingBricks.
 * @param geometry  Extent and scale of the subject.
 * @param profile   Measured densities, used to rescale the curve per level.
 * @param transfer  The level-0 curve. Coarser levels are derived from it, once per call.
 * @param eye       Camera.
 * @param params    Marching parameters.
 * @param pixels    Destination, @p width * @p height entries, 0xAARRGGBB.
 * @param width     Framebuffer width in pixels.
 * @param height    Framebuffer height in pixels.
 * @param rowFirst  First row to render, and @p rowCount how many -- so a thread pool can split a
 *                  frame by bands without this function knowing anything about threads.
 */
MarchReport march(const BrickMosaic &mosaic, const VolumeGeometry &geometry, const DensityProfile &profile,
                  const TransferFunction &transfer, const Eye &eye, const MarchParams &params, core::u32 *pixels,
                  core::u32 width, core::u32 height, core::u32 rowFirst, core::u32 rowCount) noexcept;

/**
 * @brief Folds a rendered frame into one number.
 *
 * @warning **A regression signature, not a determinism gate.** It is folded from float arithmetic,
 * so it holds across runs of one binary and is not promised across compilers or targets. This
 * project reserves the word "parity" for folds that survive that crossing, and calling this one
 * parity would put a number in a gate that cannot keep its promise.
 */
[[nodiscard]] core::u32 foldFrame(const core::u32 *pixels, core::u32 count) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_RAYMARCH_HPP
