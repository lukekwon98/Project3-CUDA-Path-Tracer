#include <optix.h>
#include <stdio.h>
#include <cuda_runtime.h> //helps intellisense
#include <optix_device.h> //GPU side optix functions, including optixTrace
#include "optixLaunchParams.h"
#include "sceneStructs.h" // Members of Pathsegment and ShadeableIntersection
#include <float.h>
#include <glm/glm.hpp>

#define TRIANGLE_TEST 0
#define SETUP_TEST 0

// OptiX supplies the variable's contents when it launches
// extern "C" preserves the exact symbol name - params
extern "C" {
	__constant__ LaunchParams params;
}

// extern "C" preserves the function name for lookup by OptiX - no mangling
// __raygen__ is the required prefix for an OptiX ray-generation program
extern "C" __global__ void __raygen__rg() {
	// Reads paths[index].ray and calls optixTrace
	// The launch is one-dimensional: each x index handles one active path
	const unsigned int index = optixGetLaunchIndex().x; //like thread index?

	if (index >= params.numPaths) {
		return;
	}

	const Ray& ray = params.paths[index].ray;

	const float3 origin = make_float3(ray.origin.x, ray.origin.y, ray.origin.z);
	const float3 direction = make_float3(ray.direction.x, ray.direction.y, ray.direction.z);

	//Traces 1 ray against the geometry in the GAS
	optixTrace(
		params.gasHandle,
		origin,
		direction,
		0.001f, // Ignore intersections extremely near the origin
		FLT_MAX, //Maximum ray parameter
		0.0f, //No motion blur
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0, //Hitgroup SBT offset
		1, //Hitgroup SBT stride
		0 //Miss record index
	);


	// Test 1 triangle
#if TRIANGLE_TEST
	//Test triangle lies on z = 0, the ray starts at 0,0,1 and travels toward negative Z, intersecting the triangle at 0,0,0

	const float3 origin = make_float3(0.0f, 0.0f, 1.0f);
	const float3 direction = make_float3(0.0f, 0.0f, -1.0f);

	printf("Tracing test ray\n");

	optixTrace(
		params.gasHandle, //Acceleration structure to traverse
		origin, // Ray starting position
		direction, // Ray direction
		0.001f, // Minimum accepted ray parameter t
		100.0f, // Maximum
		0.0f, // Ray time, no motion blur
		OptixVisibilityMask(255), // Enable all visibility mask bits
		OPTIX_RAY_FLAG_NONE, // No additional ray flags
		0, // Hitgroup SBT offset
		1, // Hitgroup SBT stride in records
		0 // Miss record index
	);

	printf("Test ray finished\n");
#endif

	// Setup Test
#if SETUP_TEST
	//Verify that raygen received the handle built on the host
	printf("GPU received GAS handle: %llu\n", static_cast<unsigned long long>(params.gasHandle));
#endif
}

// Runs when traversal finds the closest accepted intersection
extern "C" __global__ void __closesthit__ch() {
	const unsigned int index = optixGetLaunchIndex().x; //index'th thread(ray)
	ShadeableIntersection& result = params.intersections[index]; //index'th intersection

	result.t = optixGetRayTmax();

	const unsigned int primitiveIndex = optixGetPrimitiveIndex(); //got index of triangle that was hit
	const uint3 triangle = params.triangles[primitiveIndex];  //the triangle the ray hit, has 3 vertex indices, not 3 positions

	const float3 a = params.vertices[triangle.x]; 
	const float3 b = params.vertices[triangle.y];
	const float3 c = params.vertices[triangle.z];

	const glm::vec3 p0(a.x, a.y, a.z);
	const glm::vec3 p1(b.x, b.y, b.z);
	const glm::vec3 p2(c.x, c.y, c.z);

	//Geometric normal perpendicular to the triangle
	glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));

	//Orient normal against incoming ray
	const float3 direction = optixGetWorldRayDirection();
	const glm::vec3 rayDirection(direction.x, direction.y, direction.z);

	if (glm::dot(normal, rayDirection) > 0.0f) {
		normal = -normal;
	}

	result.surfaceNormal = normal;

	//Temporary: all box triangles use renderer material 0
	result.materialId = 0;

}

// Runs when traversal finds no intersection
extern "C" __global__ void __miss__ms() {
	const unsigned int index = optixGetLaunchIndex().x;
	ShadeableIntersection& result = params.intersections[index];

	result.t = -1.0f;

	// No material or surface exists
	result.materialId = -1;
	result.surfaceNormal.x = 0.0f;
	result.surfaceNormal.y = 0.0f;
	result.surfaceNormal.z = 0.0f;
	
	//printf("Ray missed\n");
}