#include <optix.h>


// extern "C" preserves the function name for lookup by OptiX - no mangling
// __raygen__ is the required prefix for an OptiX ray-generation program
extern "C" __global__ void __raygen__rg() {
	printf("Hello from OptiX raygen on the GPU\n");
}