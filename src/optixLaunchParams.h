#pragma once

#include <optix.h>
#include <vector_types.h> //float3 and uint3

// Launch parameters are data supplied for a launch, accessible to its GPU programs.
// Basically a struct of input data that we provide to the GPU programs

//Forward declarations - pointers only need the type names
struct PathSegment;
struct ShadeableIntersection;
struct Material;

// Shared layout used by CPU code and OptiX GPU programs
// The handle identifies the built acceleration structure. It doesn't contain the triangles or the GAS itself, only their GPU allocations
struct LaunchParams {
	OptixTraversableHandle gasHandle;

	// Addresses of existing GPU arrays owned by the CUDA renderer
	const PathSegment* paths;
	ShadeableIntersection* intersections;

	// Number of active paths for this launch
	unsigned int numPaths;

	const float3* vertices;
	const float3* normals;
	const uint3* triangles;
	const int* triangleMaterialIds;

	// Testing raygen loop implementation (everything runs on optix kernel)
	PathSegment* outputPaths; // Write each completed path back for finalGather()
	const Material* materials; //read material colors and emission during shading
	int iteration; //seed the random generator for the current sample
};