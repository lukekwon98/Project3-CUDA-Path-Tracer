#pragma once

#include <vector>
#include "gltfLoader.h"

struct PathSegment;
struct ShadeableIntersection;

// intiialize optix and create its device context
void initOptixContext(const std::vector<MeshData>& meshes, int lightMaterialId);

// destroy optix
void destroyOptixContext();

void launchOptixIntersections(const PathSegment* paths, ShadeableIntersection* intersections, int numPaths);

struct Material;

//NOOOOPE
void launchOptixPaths(
    PathSegment* paths,
    ShadeableIntersection* intersections,
    const Material* materials,
    int numPaths,
    int iteration);