#pragma once

#include <vector>
#include "gltfLoader.h"

struct PathSegment;
struct ShadeableIntersection;
struct Camera;

// intiialize optix and create its device context
void initOptixContext(const std::vector<MeshData>& meshes, int lightMaterialId);

// destroy optix
void destroyOptixContext();

void launchOptixIntersections(const PathSegment* paths, ShadeableIntersection* intersections, int numPaths);

struct Material;

//Bulk moving to raygen
void launchOptixPaths(
    PathSegment* paths, ShadeableIntersection* intersections,
    const Material* materials,
    int numPaths, int iteration,
    const Camera& camera, int maxBounces, bool generateCameraRays);