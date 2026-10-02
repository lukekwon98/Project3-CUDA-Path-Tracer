#include <optix.h>
#include <stdio.h>
#include <cuda_runtime.h> //helps intellisense
#include <optix_device.h> //GPU side optix functions, including optixTrace
#include "optixLaunchParams.h"
#include "sceneStructs.h" // Members of Pathsegment and ShadeableIntersection
#include <float.h>
#include <glm/glm.hpp>
#include <thrust/random.h>
#include "intersections.h"
#include "utilities.h"

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

	// ADD
	if (params.paths[index].remainingBounces <= 0) {
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
}

// Runs when traversal finds the closest accepted intersection
extern "C" __global__ void __closesthit__ch() {
	const unsigned int index = optixGetLaunchIndex().x; //index'th thread(ray)
	ShadeableIntersection& result = params.intersections[index]; //index'th intersection

	result.t = optixGetRayTmax(); //noice

	const unsigned int primitiveIndex = optixGetPrimitiveIndex(); //got index of triangle that was hit
	const uint3 triangle = params.triangles[primitiveIndex];  //the triangle the ray hit, has 3 vertex indices, not 3 positions

	const float3 a = params.vertices[triangle.x]; //triangle holds indices to each vertex, x y z are each indices of a vertex 
	const float3 b = params.vertices[triangle.y];
	const float3 c = params.vertices[triangle.z];

	const glm::vec3 p0(a.x, a.y, a.z);
	const glm::vec3 p1(b.x, b.y, b.z);
	const glm::vec3 p2(c.x, c.y, c.z);

	//Geometric normal perpendicular to the triangle
	glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));

	const float3 na = params.normals[triangle.x];
	const float3 nb = params.normals[triangle.y];
	const float3 nc = params.normals[triangle.z];

	const glm::vec3 n0(na.x, na.y, na.z);
	const glm::vec3 n1(nb.x, nb.y, nb.z);
	const glm::vec3 n2(nc.x, nc.y, nc.z);

	//Use vertex noramls only when all three are present
	if (glm::dot(n0, n0) > 0.0f && glm::dot(n1, n1) > 0.0f && glm::dot(n2, n2) > 0.0f) {
		const float2 bary = optixGetTriangleBarycentrics();

		const float w0 = 1.0f - bary.x - bary.y;
		const float w1 = bary.x;
		const float w2 = bary.y;

		const glm::vec3 interpolatedNormal = w0 * n0 + w1 * n1 + w2 * n2;

		if (glm::dot(interpolatedNormal, interpolatedNormal) > 0.0f) {
			normal = glm::normalize(interpolatedNormal);
		}
	}

	//Forced flat shading
	//normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
	//Orient normal against incoming ray
	const float3 direction = optixGetWorldRayDirection();
	const glm::vec3 rayDirection(direction.x, direction.y, direction.z);

	if (glm::dot(normal, rayDirection) > 0.0f) {
		normal = -normal;
	}

	result.surfaceNormal = normal;

	//Triangle render materials
	result.materialId = params.triangleMaterialIds[primitiveIndex];
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


//Raygen Loop Implementation
static __forceinline__ __device__
thrust::default_random_engine raygenMakeRandomEngine(int iteration, int pixelIndex, int remainingBounces) {
    // Same as the CPU-loop
    int h = utilhash((1 << 31) | (remainingBounces << 22) | iteration) ^ utilhash(pixelIndex);
    return thrust::default_random_engine(h);
}

static __forceinline__ __device__
glm::vec3 raygenSampleHemisphere( glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));

    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

//__forceinline__ tells the compiler to inline the function into the caller rather than use an ordinary function call
static __forceinline__ __device__
void raygenShadeFakeMaterial(
    PathSegment& path,
    ShadeableIntersection& intersection)
{
    if (intersection.t > 0.0f) {

        thrust::default_random_engine rng = raygenMakeRandomEngine(params.iteration, path.pixelIndex, path.remainingBounces);

        Material material = params.materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        if (material.emittance > 0.0f) {
            path.color *= material.color * material.emittance;
            path.remainingBounces = 0;
            return;
        }
        else {

            glm::vec3 intersectionPoint = getPointOnRay(path.ray, intersection.t);

            // scatterRay, would separating this optimize further?
            glm::vec3 nextDirection = raygenSampleHemisphere(intersection.surfaceNormal, rng);
            path.ray.origin = intersectionPoint;
            path.ray.direction = nextDirection;
            path.color *= material.color;
            path.remainingBounces--;

            // Bounce limit
            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }
        }
    }
    else {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return;
    }
}


static __forceinline__ __device__
PathSegment generateRaygenCameraPath(unsigned int index)
{
    RaygenCamera& cam = params.camera;

    int pixelIndex = (int)(index);
    int x = pixelIndex % cam.width;
    int y = pixelIndex / cam.width;

    thrust::default_random_engine rng = raygenMakeRandomEngine(params.iteration, pixelIndex, 0);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float xJitter = u01(rng);
    float yJitter = u01(rng);

    //restriction due to initialization of global __constant__ variable
    const glm::vec3 position(cam.position.x, cam.position.y, cam.position.z);
    const glm::vec3 view(cam.view.x, cam.view.y, cam.view.z);
    const glm::vec3 up(cam.up.x, cam.up.y, cam.up.z);
    const glm::vec3 right(cam.right.x, cam.right.y, cam.right.z);

    PathSegment path{};

    path.ray.origin = position;
    path.ray.direction = glm::normalize(view //normal pinhole camera ray jittered within pixel
        - right * cam.pixelLength.x * ((float)(x + xJitter) - (float)(cam.width) * 0.5f)
        - up * cam.pixelLength.y * ((float)(y + yJitter) - (float)(cam.height) * 0.5f));

    //thin-lens DOF
    if (cam.apertureRadius > 0.0f && cam.focalDistance > 0.0f) {
        //glm::vec3 forward = glm::normalize(view);

        //focal lane is a flat plane perpendicular to the camera's forward dir, at distance focalLength
        //ray direction is usually not parallel to foward, so the distance along the ray to the plane is longer than focal length
        //so project direction onto forward, and divide by cos angle to get distance along the ray
        float tFocus = cam.focalDistance / glm::dot(path.ray.direction, view); //find focus point

        //we know all rays should converge at this point in the focus plane, the pinhole ray just becomes an anchor
        glm::vec3 focusPoint = path.ray.origin + tFocus * path.ray.direction;

        float radius = cam.apertureRadius * sqrtf(u01(rng)); //sample uniform var then multiply by radius to get random point on aperture
        float angle = 6.28318530718f * u01(rng); //radians for polar
        float lensX = radius * cosf(angle);
        float lensY = radius * sinf(angle);

        //consider skipping this
        //glm::vec3 lensRight = glm::normalize(right);
        //glm::vec3 lensUp = glm::normalize(glm::cross(lensRight, forward));

        path.ray.origin = position + lensX * right + lensY * up; //update origin to a point on the physical lens
        path.ray.direction = glm::normalize(focusPoint - path.ray.origin); //update the direction
    }

    path.color = glm::vec3(1.0f);
    path.pixelIndex = pixelIndex;
    path.remainingBounces = params.maxBounces;

    return path;
}

extern "C" __global__ void __raygen__pathtrace()
{
    const unsigned int index = optixGetLaunchIndex().x;

    if (index >= params.numPaths) {
        return;
    }

    // Read once, all bounce updates operate on this local path.
    PathSegment path;

    // toggle
    if (params.generateCameraRays != 0) {
        path = generateRaygenCameraPath(index);
    }
    else {
        path = params.paths[index];
    }

    while (path.remainingBounces > 0) {
        const float3 origin = make_float3(path.ray.origin.x,path.ray.origin.y,path.ray.origin.z);
        const float3 direction = make_float3(path.ray.direction.x,path.ray.direction.y,path.ray.direction.z);

        optixTrace(
            params.gasHandle,
            origin,
            direction,
            0.001f,
            FLT_MAX,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0);

        // Existing CH/MS programs write this invocation's slot.
        // optixTrace returns after the selected program completes.
        ShadeableIntersection intersection = params.intersections[index];

        raygenShadeFakeMaterial(path, intersection);
    }

    // Preserve pixelIndex for the existing finalGather kernel.
    params.outputPaths[index] = path;
}
