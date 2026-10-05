# Changelog

What changed in each release, generated from the commit titles on `main`. Regenerate it with
`tools/changelog.sh` from [MasterLaplace/.github](https://github.com/MasterLaplace/.github);
an edit by hand is lost at the next release, whose check refuses a file that differs from
what the history gives.

## [0.2.0] - 2026-10-05

### Added

- **config**: Let LplPlugin states its version in config.h (#375)
- **bench**: Lpl-benchmark measures the volume renderer (#209)
- **voxel**: A minimap that shows where the eye is in the whole subject (#208)
- **voxel**: Trace a sheet the samples imply into a surface the marcher draws (#207)
- **zarr**: Read zarr v2 arrays through an injected key-value store (#206)
- **voxel**: Direct volume rendering of a bounded set of resident bricks (#205)
- **bench**: Lpl-benchmark runs one section with --only (#204)
- **bench**: Every benchmark row reports package energy per repetition (#132)
- **glclient**: Choose the screen, the corner, or the whole display
- **apps,samples**: The ring-0 world on a desktop, and a century of the Mani
- **procgen,engine**: A road across a world is planned twice, and the summary is the mean
- **procgen**: A planet is searched twice, because it cannot be searched once
- **math,procgen**: A world that stands on measured earth and closes on itself
- **engine**: The corpus makes somebody walk, and the walk comes back as history
- **history**: Time is measured in days, so precision is the width of an interval
- Worldview.lplscene actually asks for the horizon it now can
- A false horizon — terrain bends away with distance and altitude
- Derive FOV from a physical viewing distance, through CORDIC
- A carried lamp for the dark, and M to warp to the nearest cave
- Walkable caves under the streamed world, plus the sea and streamed landmarks
- Agent/, codec/, rosetta/, history/ et les gates P11 a P13
- Add terrain shaping parameters and profiling for rendering phases
- Enhance character controller with authoritative state and jump mechanics
- Add tests and implementations for view profiles, octree culling, and irradiance
- Add new tests for procedural generation and rendering
- Integrate living recipes into game pack system
- Add TerrainWorld class for procedural terrain generation and ecology simulation
- Implement WorldRecipe functionality and enhance testing framework
- Implement strict acked-baseline and precision LOD for entity state updates
- Add comprehensive tests for bitstream quantization, desync detection, entity delta codec, snapshot interpolation, lag compensation, and server mesh functionality
- Enhance session management and add tests for AOI, reconciliation, and session lifecycle
- Implement transport batching and server routing tests
- Introduce IMemoryBackend interface and its platform-specific implementations
- **network**: Refactor network address handling with Endpoint class
- **CubePile**: Manage CubePile state in static storage for deterministic simulation
- **editor**: Implement headless editing model with EditorSession and reflection-driven operations
- **procgen**: Implement deterministic procedural generation with Fixed32 value noise
- Add serialisation and deserialisation of .lplscene documents for ECS
- Refactor CubePile simulation to utilize ECS and Fixed32 components
- Add lpl-bench module for micro-benchmarking and system introspection
- Add system information gathering for benchmark context
- CubePile sample sim + GpuPhysicsBackend into physics module
- **p6**: Topology, software ray tracing, PBR+HDRI, immutable command buffers + late-latching, foveated rasterizer
- **p5**: Multi-viewport composite + render-to-texture
- **p5**: Classical lighting (Lambert/Phong/Blinn-Phong)
- **p5**: Integer-deterministic textures + textured rasterizer
- **p5**: SoA instancing + frustum culling
- **p5**: Depth-buffered software 3D rasterizer
- **p5**: Deterministic 3D camera/projection + render parity
- **scene**: Deterministic 2D scene graph (transforms + undo/redo)
- **image**: Portable PPM (P6) import/export
- **image**: BlitToFramebuffer (Image -> linear scanout)
- **image**: Deterministic 2D Painter (lines, rects, circles, blit)
- **image**: Deterministic 2D image module (Color + Image)
- **p4**: Drive KernelDisplayRenderer from a shared lpl::render::Mesh
- **render**: Add KernelDisplayRenderer — software rasteriser over IDisplayBackend
- Inject IPlatform into the Engine (DI seam) — P2b
- Add hosted Linux platform backends (lpl::platform) — P2b
- Add platform HAL interfaces + kernel backends (lpl::platform) — P2
- Port physics (collision/octree/CPU backend) to freestanding kernel
- Dispatch ECS SystemScheduler over a job-system seam (freestanding)
- Route ECS storage core + allocators through lpl/std umbrellas (freestanding)
- Add freestanding kernel target via lpl/std umbrella
- Extend linter to format CUDA files in addition to C++ files
- Add CUDA physics kernel support and enhance kernel module management
- Implement Vulkan SwapChain and Wrapper classes
- Implement PacketQueue and SessionManager for network event handling
- : add management of the abstract BCI source and calibration in the client
- Add BCI sources and calibration functionality
- Refactor network and system scheduling for improved clarity and performance
- Reorganize member initialization in OpenBCIDriver and remove redundancy from #pragma once
- Refactor entity management and network packet handling for improved clarity and consistency
- Add NeuralControl structure and integrate into Partition and SystemScheduler
- Enhance data reading and FFT processing in OpenBCIDriver
- Implement Fast Fourier Transform and enhance NeuralState management in OpenBCIDriver
- Improve concentration calculation and add channel parsing in OpenBCIDriver
- Add the OpenBCIDriver class for BCI data management
- Add Android client implementation with OpenGL ES support
- Add socket fallback for visual client test
- Add size method to Network class and update client count logging in main
- Enhance physics system with sleeping entities and multi-threading support
- Fix the active state check in ThreadPool and improve the force calculations in Partition
- Add a ThreadPool for managing parallel tasks in the scheduler
- Add visual 3D demo with interactive camera and OpenGL integration
- Add methods for entity management and improve migration simulation between chunks
- Add the FlatDynamicOctree class and improve entity management with references and query methods
- Add entity movement simulation with chunk management
- Enhance partition management with chunk creation and entity migration handling
- Add the WorldPartition class for managing spatial partitions
- Implement vector and quaternion math structures, along with a partitioning system for spatial management
- Add a high-performance, lock-free hash map for world partitioning
- Add Morton encoding utilities and implement a World Partitioning system
- Add open and release functions for buffer memory management
- Add a hook to intercept UDP packets on port 7777
- Refactor core structure for entity management and add GPU physics update kernel
- Implement GPU-based physics update and expose run_physics_gpu function
- Add plugin.h header file with network structures and API
- Add initial implementation of ECS with CUDA support
- Implement network simulation and game loop in main.c
- Implement network entity management with ring buffer

### Fixed

- **config**: The header is lplplugin/config.h, as its users expect (#377)
- **bench**: The energy column reports nothing it cannot stand behind (#358)
- **ci**: The linter pushes to the branch it formatted, and the commit check reaches every branch and pull request (#4)
- **rosetta**: Store a mnemonic at the width of the field it is written into
- **apps**: Open on the monitor somebody is sitting in front of
- **apps**: A raw-Xlib window has to name itself or WSLg never shows it
- **glclient**: Check the buffer the window receives, not the one the engine wrote
- Bind the cave warp to the key an AZERTY keyboard labels M, and name unbound keys
- Send a keyboard turn to the body, not to the camera the body overwrites
- Remove duplicate getter methods in Config class
- Update references from kernel_std to kstd in header files
- **image**: Use clear()+resize() so Image compiles freestanding
- Add missing build deps (glfw, glm, vulkan-loader) + auto shader compilation
- Refactor BCI plugin directory structure and update include paths
- Add build directories to .gitignore
- Adjust FFT output size and improve pipeline error handling
- Update README and code comments for clarity and consistency; add normalize method to Vec3
- Remove the duplicate definition and implementation of checkAndMigrate
- Correct the management of entity identifiers and adjust the generation logic
- Correct boundary check logic and mass assignment in physicsTick
- Correct entity index handling in physicsTick and improve entity snapshot creation
- Improve error handling and add performance metrics in the kernel and main modules
- Improve error handling and add performance metrics in the kernel and main modules
- Add personal files to .gitignore and improve package management in the module
- Update Makefile for simplified build and run commands
- Update variable visibility and improve code structure in lpl_kmod.c
- Add initializations and management functions for the kernel module in lpl_kmod.c and main.c
- Correct circular buffer management and improve kernel binding in main.c and plugin.cu
- Improve UDP packet handling and buffer management in lpl_kmod.c and plugin.h
- Correct data types and improve packet handling in lpl_kmod.c and plugin.h
- Correct data types and improve error handling in packet interception and memory mapping functions
- Update memory allocation and error handling in lpl_open and lpl_release functions
- Remove early return condition for negative y values in physics update kernel
- Update entity index usage in rendering output for correct position display
- Rename plugin to plugin.c

### Performance

- **concurrency**: Spinlock backs off exponentially while the lock is held (#130)

### Changed

- **concurrency**: Lock() says what a caller must know, under test (#357)
- Enhance platform and input management
- **render,math**: Make the render portable-core freestanding
- **engine**: Route Config strings through lpl/std
- **engine**: Reparent GameLoop onto platform::IClockBackend
- Update header guards and access specifiers across multiple files
- Remove obsolete physics and rendering interfaces, implementations, and related files
- Create a complete architecture based on previous work
- Update BCI plugin integration and enhance build process in Makefile
- Delete lpl/bci folder and update aadd_repositories to use forked one
- Recreate bci plugin and add unit tests for BCI module components
- Move bci in plugins
- Create a plugin architecture and add bci signal metrics + riemannian geometry
- Replace direct socket communication with a kernel driver-based shared memory approach
- Update visual3d.cpp: Enhance network handling, camera control, and entity synchronization
- Enhance entity management and network communication in the Laplacian engine
- Restructure README for clarity and update project overview
- Update Morton encoding/decoding utilities and improve Radix Sort implementation
- Add documentation comments and reorganize the code in the lpl_kmod.c and plugin.cu files
- Improve code readability and structure by adjusting declaration styles and adding comments

### Documentation

- **contributing**: Say what holds for code the kernel compiles (#371)
- Update README with project status and legacy info
- Add bci diagram
- Change old documentation, visual3d to visual
- Update architecture diagram and enhance networking section in README.md
- Add a table of contents to README.md
- Update the documentation to reflect the progress of the Kernel module development
- Correct the name of the main engine file and update the progress status in the roadmap
- Add build and run instructions to README
- Add project roadmap and objectives to README

### Housekeeping

- **license**: Move to MIT, VkWrapper included, and add a citation file (#373)
- Remove deprecated repository reference for brainflow-liblsl
- Remove apt cache from repo
- Delete unnecessary cache files
- Remove build instructions comment from visual.cpp
- Update README.md, delete server.cpp and rename visual3d to visual
- Delete the obsolete plugin.c file
- Refactor server functions for better modularity
- Refactor plugin.cu for C++ and CUDA compatibility
- Refactor network thread and main loop for clarity

### Other

- Schumcher formula example
- Remove Morton encoding utilities, radix sort implementation, and world partitioning system (legacy code)
