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
	glm::vec3 geometricNormal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    glm::vec3 normal = geometricNormal;

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

    result.geometricNormal = geometricNormal;

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
    result.geometricNormal = glm::vec3(0.0f);

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

static __forceinline__ __device__
float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    //positive incident cosine and the correct incident/transmitted indices of refraction
    //cosThetaI = cos angle between incoming light and surface normal
    cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);

    if (etaI == etaT) {
        return 0.0f;
    }

    float eta = etaI / etaT;
    float sin2ThetaI = glm::max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(1.0f - sin2ThetaT);

    float rParallel = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);
    float rPerpendicular = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);

    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

static __forceinline__ __device__
float TrowbridgeReitzD(float NoH, float roughness) { //NoH: surface normal with microfacet normal or halfway direction
    if (NoH <= 0.0f) { //NoH = dot(normal, microfacetNormal)
        return 0.0f;
    }

    float cos2Theta = NoH * NoH;
    float denominator = (1.0f - cos2Theta) + roughness * roughness * cos2Theta;

    return roughness * roughness / (PI * denominator * denominator);
}

static __forceinline__ __device__
float Lambda(float NoV, float roughness) {  //NoV: surface normal with direction previous path vertex
    float cos2Theta = NoV * NoV;
    float sin2Theta = glm::max(0.0f, 1.0f - cos2Theta);
    float tan2Theta = sin2Theta / cos2Theta;

    float alpha2Tan2Theta = roughness * roughness * tan2Theta;

    return (-1.0f + sqrtf(1.0f + alpha2Tan2Theta)) / 2.0f;
}


static __forceinline__ __device__
float TrowbridgeReitzG(float NoV, float NoL, float roughness) { //NoL: surface normal with newlyl sampled outgoing direction
    if (NoV <= 0.0f || NoL <= 0.0f) {
        return 0.0f;
    }

    return 1.0f / (1.0f + Lambda(NoV, roughness) + Lambda(NoL, roughness));
}

//Randomly chooses the normal of a tiny mirror on a rough surface
//note that xi is a random number generator
static __forceinline__ __device__
glm::vec3 Sample_wh(const glm::vec3& nor, const glm::vec2& xi, float roughness) {
    float phi = TWO_PI * xi.y; //chooses a direction around the surface normal

    float tanTheta2 = roughness * roughness * xi.x / (1.0f - xi.x); //chooses how much the microfacet tilts away from the surface normal, following GGX distribution. small roughness -> concnetrates near nor, larger -> spreads them wider

    //spherial-coordinate direction, local
    float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
    float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 wh(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

    //helper axis, checking if helper axis is 
    glm::vec3 directionNotNormal = fabsf(nor.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);

    glm::vec3 tangent = glm::normalize(glm::cross(directionNotNormal, nor));
    glm::vec3 bitangent = glm::cross(nor, tangent);

    //rotate direction to match actual surface
    return glm::normalize(wh.x * tangent + wh.y * bitangent + wh.z * nor);
}

static __forceinline__ __device__
float TrowbridgeReitzPdf(const glm::vec3& nor, const glm::vec3& wh, float roughness) {
    float NoH = glm::clamp(glm::dot(nor, wh), 0.0f, 1.0f);

    return TrowbridgeReitzD(NoH, roughness) * NoH;
}

static __forceinline__ __device__
glm::vec3 fresnel(float cosTheta, const glm::vec3& R, float roughness){
    cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);

    return R + (glm::max(glm::vec3(1.0f - roughness), R) - R) * powf(1.0f - cosTheta, 5.0f);
}

static __forceinline__ __device__
glm::vec3 f_microfacet_refl(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec3& wo, const glm::vec3& wi, float roughness) {
    float cosThetaO = glm::clamp(glm::dot(nor, wo), -1.0f, 1.0f);
    float cosThetaI = glm::clamp(glm::dot(nor, wi), -1.0f, 1.0f);

    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = wo + wi;

    if (glm::dot(wh, wh) == 0.0f) {
        return glm::vec3(0.0f);
    }

    wh = glm::normalize(wh);

    float NoH = glm::clamp(glm::dot(nor, wh), 0.0f, 1.0f);

    glm::vec3 F = fresnel(glm::dot(wo, wh), albedo, roughness);
    float D = TrowbridgeReitzD(NoH, roughness);
    float G = TrowbridgeReitzG(cosThetaO, cosThetaI, roughness);

    return D * G * F / (4.0f * cosThetaI * cosThetaO);
}

static __forceinline__ __device__
glm::vec3 f_diffuse(const glm::vec3& albedo) {
    return albedo / PI;
}

static __forceinline__ __device__
glm::vec3 f_metallic_roughness(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec3& wo,
    const glm::vec3& wi, float roughness, float metallic) {
    glm::vec3 N = nor;

    float NdotV = glm::clamp(glm::dot(N, wo), 0.0f, 1.0f);
    float NdotL = glm::clamp(glm::dot(N, wi), 0.0f, 1.0f);

    if (NdotV <= 0.0f || NdotL <= 0.0f || roughness <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = wo + wi;

    if (glm::dot(wh, wh) == 0.0f) {
        return glm::vec3(0.0f);
    }

    wh = glm::normalize(wh);
    metallic = glm::clamp(metallic, 0.0f, 1.0f);

    glm::vec3 R = glm::mix(glm::vec3(0.04f), albedo, metallic);
    glm::vec3 F = fresnel(glm::dot(wo, wh), R, roughness);

    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0f) - kS;
    kD *= 1.0f - metallic;

    glm::vec3 diffuse = f_diffuse(albedo);

    float NoH = glm::clamp(glm::dot(N, wh), 0.0f, 1.0f);
    float D = TrowbridgeReitzD(NoH, roughness);
    float G = TrowbridgeReitzG(NdotV, NdotL, roughness);

    glm::vec3 specular = D * G * F / (4.0f * NdotV * NdotL);

    return kD * diffuse + specular;
}

static __forceinline__ __device__
glm::vec3 Sample_f_metallic_roughness(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec3& wo, float roughness, float metallic,
    thrust::default_random_engine& rng, glm::vec3& wiW, float& pdf) {

    wiW = glm::vec3(0.0f);
    pdf = 0.0f;

    if (glm::dot(nor, wo) <= 0.0f || roughness <= 0.0f) {
        return glm::vec3(0.0f);
    }

    metallic = glm::clamp(metallic, 0.0f, 1.0f);

    // Sampling probability
    float pSpecular = 0.5f + 0.5f * metallic;

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float choose = glm::min(u01(rng), 0.99999994f);
    glm::vec3 wi;

    if (choose < pSpecular) {
        glm::vec2 xi;
        xi.x = glm::min(u01(rng), 0.99999994f);
        xi.y = u01(rng);

        glm::vec3 wh = Sample_wh(nor, xi, roughness);

        if (glm::dot(wo, wh) <= 0.0f) {
            return glm::vec3(0.0f);
        }
        wi = glm::normalize(glm::reflect(-wo, wh));
    }
    else {
        wi = glm::normalize(raygenSampleHemisphere(nor, rng));
    }

    float cosThetaI = glm::clamp(glm::dot(nor, wi), 0.0f, 1.0f);
    if (cosThetaI <= 0.0f) {
        return glm::vec3(0.0f);
    }

    // Compute both sampling densities for the chosen direction
    glm::vec3 wh = wo + wi;
    if (glm::dot(wh, wh) == 0.0f) {
        return glm::vec3(0.0f);
    }

    wh = glm::normalize(wh);
    float woDotWh = glm::dot(wo, wh);
    if (woDotWh <= 0.0f) {
        return glm::vec3(0.0f);
    }

    float pdfSpecular = TrowbridgeReitzPdf(nor, wh, roughness) / (4.0f * woDotWh);
    float pdfDiffuse = cosThetaI / PI;

    pdf = pSpecular * pdfSpecular + (1.0f - pSpecular) * pdfDiffuse;

    wiW = wi;

    return f_metallic_roughness(albedo, nor, wo, wi, roughness, metallic);
}

static __forceinline__ __device__
glm::vec3 Sample_f_microfacet_refl(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec2& xi, const glm::vec3& wo, float roughness, glm::vec3& wiW, float& pdf) {
    wiW = glm::vec3(0.0f);
    pdf = 0.0f;

    if (glm::dot(nor, wo) <= 0.0f || roughness <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = Sample_wh(nor, xi, roughness);

    float woDotWh = glm::dot(wo, wh);
    if (woDotWh <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wi = glm::normalize(glm::reflect(-wo, wh));

    //Reflection must stay above shading surface
    if (glm::dot(nor, wi) <= 0.0f) {
        return glm::vec3(0.0f);
    }

    pdf = TrowbridgeReitzPdf(nor, wh, roughness) / (4.0f * woDotWh);

    wiW = wi;

    return f_microfacet_refl(albedo, nor, wo, wi, roughness);
}

//__forceinline__ tells the compiler to inline the function into the caller rather than use an ordinary function call
// 1: pure refractive
// 2: metal/roughness (when 0, still diffuse + specular (plastic), more metallic means specular color moves towards the base color and removes diffuse)
// 3: pure reflective (+ roughness)
// 4: pure diffuse
static __forceinline__ __device__
void raygenShadeFakeMaterial(PathSegment& path, ShadeableIntersection& intersection) {
    if (intersection.t > 0.0f) {

        thrust::default_random_engine rng = raygenMakeRandomEngine(params.iteration, path.pixelIndex, path.remainingBounces);

        Material material = params.materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        if (material.emittance > 0.0f) {
            path.color *= material.color * material.emittance;
            path.remainingBounces = 0;
            return;
        }
        else if (material.hasRefractive > 0.0f) {
            glm::vec3 incident = glm::normalize(path.ray.direction);
            bool frontFace = glm::dot(incident, intersection.geometricNormal) < 0.0f;

            glm::vec3 normal = frontFace ? intersection.geometricNormal : -intersection.geometricNormal;

            float etaI = frontFace ? 1.0f : material.indexOfRefraction;
            float etaT = frontFace ? material.indexOfRefraction : 1.0f;
            float eta = etaI / etaT;

            float cosThetaI = glm::clamp(glm::dot(-incident, normal), 0.0f, 1.0f);
            float F = fresnelDielectric(cosThetaI, etaI, etaT); //probability

            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

            glm::vec3 nextDirection;

            if (F >= 1.0f || u01(rng) < F) { //F becomes the possibility to reflect/refract
                nextDirection = glm::reflect(incident, normal);
            }
            else {
                nextDirection = glm::refract(incident, normal, eta);

                path.color *= eta * eta; //pbrt's radiacne transport correction
            }

            nextDirection = glm::normalize(nextDirection);

            //exact surface position
            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;
            //move onto the side the outgoing ray is traveling toward
            float offsetSign = glm::dot(nextDirection, intersection.geometricNormal) > 0.0f ? 1.0f : -1.0f; //reflection stays in th incident medium, refraction crosses the surface

            path.ray.origin = hitPoint + offsetSign * 0.001f * intersection.geometricNormal;
            path.ray.direction = nextDirection;
            path.remainingBounces--;

            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }

        }
        //metallic 0 - colroed diffuse reflection and neutral specular reflection (plastic)
        //metallic 1 - colored specular reflection, no diffuse component (metal)
        else if (material.useMetallicRoughness != 0) {
            glm::vec3 wo = -glm::normalize(path.ray.direction);

            glm::vec3 geometricNormal = glm::normalize(intersection.geometricNormal);
            if (glm::dot(geometricNormal, wo) < 0.0f) {
                geometricNormal = -geometricNormal;
            }

            glm::vec3 nor = glm::normalize(intersection.surfaceNormal);
            if (glm::dot(nor, geometricNormal) < 0.0f) {
                nor = -nor;
            }
            if (glm::dot(nor, wo) <= 0.0f) {
                nor = geometricNormal;
            }

            float roughness = glm::clamp(material.roughness, 0.001f, 1.0f);
            float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);

            glm::vec3 wiW;
            float pdf = 0.0f;

            glm::vec3 f = Sample_f_metallic_roughness(material.color, nor, wo, roughness, metallic, rng, wiW, pdf);

            if (!(pdf > 0.0f) || glm::dot(wiW, geometricNormal) <= 0.0f) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
                return;
            }

            float cosThetaI = glm::clamp(glm::dot(nor, wiW), 0.0f, 1.0f);

            path.color *= f * cosThetaI / pdf;

            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;

            path.ray.origin = hitPoint + 0.001f * geometricNormal;
            path.ray.direction = glm::normalize(wiW);
            path.remainingBounces--;

            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }
        }
        else if (material.hasReflective > 0.0f) {
            glm::vec3 incident = glm::normalize(path.ray.direction); //world space ray
            glm::vec3 wo = -incident;

            //If shading normal reflects ray into the object, use geometric normal instead
            //Geometric normal, flip if incident is on the opposite side
            glm::vec3 geometricNormal = glm::normalize(intersection.geometricNormal);

            if (glm::dot(geometricNormal, wo) < 0.0f) {
                geometricNormal = -geometricNormal;
            }

            //Align the shading normal with the geometric normal
            glm::vec3 nor = glm::normalize(intersection.surfaceNormal);
            if (glm::dot(nor, geometricNormal) < 0.0f) {
                nor = -nor;
            }

            if (glm::dot(nor, wo) <= 0.0f) {
                nor = geometricNormal;
            }

            float roughness = glm::clamp(material.roughness, 0.0f, 1.0f);
            glm::vec3 wiW;

            if (roughness == 0.0f) {
                // Perfect mirror
                glm::vec3 wh = nor;
                wiW = glm::reflect(-wo, wh);

                if (glm::dot(wiW, geometricNormal) <= 0.0f){
                    wh = geometricNormal;
                    wiW = glm::reflect(-wo, wh);
                }
                path.color *= fresnel(glm::dot(wo, wh), material.color, 0.0f);
            }
            else {
                //Avoid numerically extreme GGX values
                roughness = glm::max(roughness, 0.001f);

                thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

                glm::vec2 xi;
                xi.x = glm::min(u01(rng), 0.99999994f); //largest representable float below 1.0f, exclusive
                xi.y = glm::min(u01(rng), 0.99999994f);

                float pdf = 0.0f;

                glm::vec3 f = Sample_f_microfacet_refl(material.color, nor, xi, wo, roughness, wiW, pdf);

                if (!(pdf > 0.0f) || glm::dot(wiW, geometricNormal) <= 0.0f) {
                    path.color = glm::vec3(0.0f);
                    path.remainingBounces = 0;
                    return;
                }

                float cosThetaI = glm::clamp(glm::dot(nor, wiW), 0.0f, 1.0f);
                path.color *= f * cosThetaI / pdf;
            }

            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;

            path.ray.origin = hitPoint + 0.001f * geometricNormal;
            path.ray.direction = glm::normalize(wiW);
            path.remainingBounces--;

            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }
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
