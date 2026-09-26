#pragma once

struct PathSegment;
struct ShadeableIntersection;

// intiialize optix and create its device context
void initOptixContext();

// destroy optix
void destroyOptixContext();

void launchOptixIntersections(const PathSegment* paths, ShadeableIntersection* intersections, int numPaths);