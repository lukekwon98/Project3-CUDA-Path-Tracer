#include <optix.h>
#include <stdio.h>
#include <cuda_runtime.h> //helps intellisense
#include <optix_device.h> //GPU side optix functions, including optixTrace
#include "optixLaunchParams.h"

// OptiX supplies the variable's contents when it launches
// extern "C" preserves the exact symbol name - params
extern "C" {
	__constant__ LaunchParams params;
}

// extern "C" preserves the function name for lookup by OptiX - no mangling
// __raygen__ is the required prefix for an OptiX ray-generation program
extern "C" __global__ void __raygen__rg() {
	//Test triangle lies on z = 0, the ray starts at 0,0,1 and travels toward negative Z, intersecting the triangle at 0,0,0
	const float3 origin = make_float3(0.0f, 0.0f, 1.0f);
	const float3 direction = make_float3(0.0f, 0.0f, -1.0f);

	printf("Tracing test ray\n");

	optixTrace(
		params.gasHandle,
		origin,
		direction,
		0.001f,
		100.0f,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,
		1,
		0
	);

	printf("Test ray finished\n");
	
	//Verify that raygen received the handle built on the host
	//printf("GPU received GAS handle: %llu\n", static_cast<unsigned long long>(params.gasHandle));
}

// Runs when traversal finds the closest accepted intersection
extern "C" __global__ void __closesthit__ch() {
	printf("Triangle hit\n");
}

// Runs when traversal finds no intersection
extern "C" __global__ void __miss__ms() {
	printf("Ray missed\n");
}