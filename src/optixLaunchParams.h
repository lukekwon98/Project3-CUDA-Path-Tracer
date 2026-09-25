#pragma once

#include <optix.h>

// Launch parameters are data supplied for a launch, accessible to its GPU programs.
// Basically a struct of input data that we provide to the GPU programs

// Shared layout used by CPU code and OptiX GPU programs
// The handle identifies the built acceleration structure. It doesn't contain the triangles or the GAS itself, only their GPU allocations
struct LaunchParams {
	OptixTraversableHandle gasHandle;
};