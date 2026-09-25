// initialization and cleanup function declarations
#include "optixSetup.h"

// CUDA Runtime API, including cudaFree()
#include <cuda_runtime.h>

// Optix APi types, including OptixDeviceContext
#include <optix.h>

// Defines the function table that holds OptiX function pointers
// Must be inluded in exactly ONE .cpp file within the application
#include <optix_function_table_definition.h>

// Provides optixInit() and wrapperes that call through that table
#include <optix_stubs.h>

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>

#include "optixLaunchParams.h"

//PTX: Intermediate GPu instructions generated from optixPrograms.cu - generated on build, not runtime (created by specifying __raygen__rg and tweaking cmakeslists)
//Module: OPtiX compiles the PTX into a module containing the GPU program
//Program group: Selects __raygen__rg from that module as the raygen program
//Pipeline: Links the selected program groups into an exectuable GPU pipeline
//SBT: Supplies records identifying which programs to invoke for a particular launch, plus optional data - it's like binding shaders to a renderer

//optixContext: Create context - identifies the shared OptiX environment where we create modules, program groups, and pipelines
//optixModule: Create module from PTX - Identifies the module containing the compiled GPU program
//pipelineCompileOptions: Configure Compilation - Stores settings shared by module and pipeline creation. Settings, not a handle
//raygenProgramGroup: Select raygen function - Identifies the program group selection __raygen__rg from optixModule
//optixPipeline: Link pipeline - identifies the executable pipeline built from that program group
//dev_raygenRecord: Upload SBT record - Points to the GPU allocation containing the packed raygen record
//sbt: Describe SBT + Launch: CPU-side structure whose raygenRecord field holds that GPU address, will be passed into launch call

namespace { // Anonymous namespace makes names private to this .cpp file
	// Handle to the OptiX device context
	// nullptr means we have not created a context yet
	OptixDeviceContext optixContext = nullptr;

	// Handle to the module containing our GPU program - moduleOptiosn controls compilation choices such as optimization
	OptixModule optixModule = nullptr;

	// Keeping settings so we can resue them when creating the pipeline
	// Module creation and pipeline creation must use consistent settings - describes features the eventual pipeline will use (motion blur, payload values, attribute values etc)
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	// optixContext - holds optiX state associated with the CUDA context. Modules and pipelines belong to it
	// optixModule - contains compiled GPU programs from the PTX (__raygen__rg())
	// pipelineCompileOptions - a settings structure describing pipeline features. isn't a context or an executeable pipeline

	// Handle to the program group selecting the raygen function - tells OptiX to use the function named __raygen__rg from this module as a ray generation program
	// It's called a group, but a raygen program grou pselects just one entry function - a hit group can combine closest-hit, any-hit, and intersection functions
	// Selects functions from a module and specifies their roles in the pipeline
	OptixProgramGroup raygenProgramGroup = nullptr;

	// Miss program group - tells OptiX which program to run when a traced ray hits no geometry
	OptixProgramGroup missProgramGroup = nullptr;

	// Handle to the executable pipeline that links the selected GPU programs, creating it does not launch GPU work
	OptixPipeline optixPipeline = nullptr;
	
	// Selects the programs used when a ray intersects our triangle
	OptixProgramGroup hitgroupProgramGroup = nullptr;

	// Temporary GPU workspace used during GAS construction
	void* dev_gasTempBuffer = nullptr;
	// GPU allocation holding the completed acceleration structure
	void* dev_gasOutputBuffer = nullptr;
	// Opaque identifier returned by Optix for the completed GAS
	OptixTraversableHandle gasHandle = 0;

	// GPU buffer holding the launch parameters supplied to OptixLaunch()
	LaunchParams* dev_launchParams = nullptr;

	// Read the generated PTX file into CPU memory
	std::string loadPtxFile(const char* path) {
		std::ifstream file(path, std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error(std::string("Could not open PTX file: ") + path);
		}

		std::string ptx{
			std::istreambuf_iterator<char>(file),
			std::istreambuf_iterator<char>()
		};

		if (ptx.empty()) {
			throw std::runtime_error(std::string("PTX file is empty: ") + path);
		}

		return ptx;
	}

	//////////////////////////////////
	// Shader binding table (SBT) - provides records identifying the programs to use during a launch, along wiht optional program-specific data
	//////////////////////////////////

	// Define the layout for the raygen SBT record
	// alignas: ensures that records satisfy OptiX's alignment requirement
	struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { //identifies which OptiX program group to invoke
		// OptiX will fill this with information identifying the program
		char header[OPTIX_SBT_RECORD_HEADER_SIZE]; //binary storage, not a text string - OptiX defines it as 32 bytes and requires SBT record alignment of 16 bytes
	};

	// Same header only layout, it will identify our hitgroup
	using HitgroupRecord = RaygenRecord;

	// GPU allocation containing the packed record
	HitgroupRecord* dev_hitgroupRecord = nullptr;

	//Header only layout - same layout as our raygen record
	using MissRecord = RaygenRecord;
	MissRecord* dev_missRecord = nullptr;

	// Pointer to teh GPU allocation that holds the raygen record, no GPU memory memory actually allocated
	RaygenRecord* dev_raygenRecord = nullptr;

	// CPU-side description of where the SBT records are stored on the GPU, zero initialized
	OptixShaderBindingTable sbt = {};

	// Detailed diagnostics reported by OptiX to the CPU
	void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
		std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << std::endl;
	}

	//GPU allocation containing the three vertices of a test triangle
	float3* dev_testVerticies = nullptr;


	//Error printing Helpers
	void checkCuda(cudaError_t result, const char* operation) {
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
		}
	}

	void checkOptix(OptixResult result, const char* operation) {
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string(operation) + ": " + optixGetErrorString(result));
		}
	}
}

void initOptixContext() {
	// Ensure Cuda is initialized for the current device, passing nullptr frees no allocation
	// Stop initialization if CUDA reports an error
	cudaError_t cudaResult = cudaFree(nullptr);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA initialization failed: ") + cudaGetErrorString(cudaResult));
	}

	//////////////////////////////////
	// Load the OptiX driver API and populate its function table
	//////////////////////////////////
	OptixResult optixResult = optixInit();

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("Optix initialization failed. Eror code: ") + std::to_string(static_cast<int>(optixResult)));
	}

	//////////////////////////////////
	// Create Optix Context
	//////////////////////////////////
	// Zero-initialize all options, these options configure the context we'll create
	OptixDeviceContextOptions options = {};

	// Enable diagnostic messages, including detailed information
	options.logCallbackFunction = optixLogCallback;
	options.logCallbackLevel = 4;

	// Enable additional checks while debugging the setup
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

	// nullptr tells OptiX to use the current CUDA context, configured with &options, and writes the created handle into &optixContext.
	// reusing existing optixREsult 
	optixResult = optixDeviceContextCreate(nullptr, &options, &optixContext);

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX context creation failed: ") + optixGetErrorString(optixResult));
	}

	std::cout << "OptiX context created." << std::endl;

	// Print the PTX path supplied by CMake
	std::cout << "OptiX PTX path: " << OPTIX_PTX_PATH << std::endl;

	// Load the intermediate GPU program using the path supplied by CMake
	std::string ptx = loadPtxFile(OPTIX_PTX_PATH);

	// Confirm we actually read the file
	std::cout << "Loaded PTX: " << ptx.size() << " bytes." << std::endl;

	// Configure compilation of this module
	OptixModuleCompileOptions moduleOptions = {};
	moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	// First test only runs raygen, does not trace rays
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.numPayloadValues = 0;
	pipelineCompileOptions.numAttributeValues = 2; // built-in triangle intersections provide 2 barycentric coordinates
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // test pipeline uses triangle geometry

	// Haven't declared a GPU laun-parameter variable yet
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

	// OptiX writes compiler diagnostics into this CPU buffer
	char log[4096] = {};
	size_t logSize = sizeof(log);

	//////////////////////////////////
	// Create Optix Module
	//////////////////////////////////
	optixResult = optixModuleCreate(optixContext, &moduleOptions, &pipelineCompileOptions, 
		ptx.c_str(), // Pointer to PTX text
		ptx.size(), // Length of the text in bytes
		log, 
		&logSize, // Buffer capacity in, log size out
		&optixModule);
	
	// Ensure printing stays within the buffer even if the log was truncated
	log[sizeof(log) - 1] = '\0';
	if (log[0] != '\0') {
		std::cout << "OptiX module log:\n" << log << std::endl;
	}

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX module creation failed: ") + optixGetErrorString(optixResult));
	}

	std::cout << "Optix module created." << std::endl;

	// Describe which GPU function we want and its role
	OptixProgramGroupDesc raygenDesc = {};
	raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; //kind is raygen
	raygenDesc.raygen.module = optixModule;
	raygenDesc.raygen.entryFunctionName = "__raygen__rg";

	// Default program-group options are sufficient for initial test
	OptixProgramGroupOptions programGroupOptions = {};

	// Reuse the diagnostic buffer. reset its capacity because the previous call changed logSize
	log[0] = '\0';
	logSize = sizeof(log);

	//////////////////////////////////
	// Create raygen program group - selects functions from a module and specifies their roles in the pipeline
	//////////////////////////////////

	optixResult = optixProgramGroupCreate(
		optixContext, 
		&raygenDesc,
		1,// Number of program groups to create
		&programGroupOptions,
		log,
		&logSize,
		&raygenProgramGroup); //receives the created handle

	log[sizeof(log) - 1] = '\0';
	if (log[0] != '\0') {
		std::cout << "OptiX program group log:\n" << log << std::endl;
	}

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX raygen program group creation failed: ") + optixGetErrorString(optixResult));
	}

	std::cout << "OptiX raygen program group created." << std::endl;

	//////////////////////////////////
	// Create miss program group
	//////////////////////////////////
	OptixProgramGroupDesc missDesc = {};
	missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	//Replace the empty miss behavior with the GPU function
	missDesc.miss.module = optixModule;
	missDesc.miss.entryFunctionName = "__miss__ms";

	optixResult = optixProgramGroupCreate(optixContext,
		&missDesc,
		1,
		&programGroupOptions,
		nullptr,
		nullptr,
		&missProgramGroup);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("Miss program group creation failed: ")+ optixGetErrorString(optixResult));
	}


	//////////////////////////////////
	// Create hit program group
	//////////////////////////////////
	OptixProgramGroupDesc hitgroupDesc = {};
	hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	//CH = closest-hit
	hitgroupDesc.hitgroup.moduleCH = optixModule;
	hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

	//AH remains null: we aren't using an any-hit program
	//IS remains null: triangles use Optix's built in intersection

	optixResult = optixProgramGroupCreate(
		optixContext,
		&hitgroupDesc,
		1,
		&programGroupOptions,
		nullptr, // the context callback supplies diagnostics
		nullptr,
		&hitgroupProgramGroup
	);
	checkOptix(optixResult, "Hitgroup program group creation failed");

	std::cout << "OptiX hitgroup program group created." << std::endl;

	//////////////////////////////////
	// Link raygen program group into pipeline
	//////////////////////////////////
		// Pipeline includes these
	OptixProgramGroup programGroups[] = {
		raygenProgramGroup,
		missProgramGroup,
		hitgroupProgramGroup
	};

	OptixPipelineLinkOptions linkOptions = {};

	// Trace recursion depth, not the rendereer's total bounce count
	linkOptions.maxTraceDepth = 1;

	log[0] = '\0';
	logSize = sizeof(log);

	optixResult = optixPipelineCreate(
		optixContext, // Context that owns the OptiX objects
		&pipelineCompileOptions, // Same compile settings for the module
		&linkOptions, // Settings for linking the pipeline
		programGroups, // Address for our single program-group handle
		3, // Number of program groups
		log, // diagnostics
		&logSize, 
		&optixPipeline // handle
	);

	log[sizeof(log) - 1] = '\0';
	if (log[0] != '\0') {
		std::cout << "OptiX pipeline log: \n" << log << std::endl;
	}
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX pipeline creation failed: ") + optixGetErrorString(optixResult));
	}
	std::cout << "OptiX pipeline created." << std::endl;

	//////////////////////////////////
	// Create a record in CPU memory
	//////////////////////////////////
	RaygenRecord raygenRecord = {}; //initially filled with zeros


	// Ask OptiX to write the program group's identifying information into the record's header
	optixResult = optixSbtRecordPackHeader(
		raygenProgramGroup, // Program group that the record will identify
		&raygenRecord // Address of the CPU record to fill
	);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX raygen record header packing failed: ") + optixGetErrorString(optixResult));
	}
	std::cout << "OptiX raygen record header packed." << std::endl; //Packing: writing OptiX's internal binary information into the header

	//////////////////////////////////
	// Copy packed raygen record to GPU memory
	//////////////////////////////////
	
	// Allocate GPU memory for one raygen record, dev_raygenRecord has the allocation's address
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_raygenRecord), sizeof(RaygenRecord));
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Raygen record allocation failed: ") + cudaGetErrorString(cudaResult));
	}

	// Copy the packed CPU record into the GPU allocation
	cudaResult = cudaMemcpy(
		dev_raygenRecord,
		&raygenRecord, // Source is our local GPU record
		sizeof(RaygenRecord),
		cudaMemcpyHostToDevice
	);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Raygen record upload failed: ") + cudaGetErrorString(cudaResult));
	}

	// Tell the SBT description where the GPU record lives, CUdeviceptr is the device-address type expected by OptiX. Converts the address representation, copies no data
	sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(dev_raygenRecord);

	std::cout << "OptiX raygen SBT record uploaded." << std::endl;

	//////////////////////////////////
	// Miss record stuff
	//////////////////////////////////

	// Prepare the miss record in CPU memory
	MissRecord missRecord = {};
	optixResult = optixSbtRecordPackHeader(
		missProgramGroup, &missRecord
	);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("Miss record packing failed: ") + optixGetErrorString(optixResult));
	}

	//Allocate GPU storage for this record
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_missRecord), sizeof(MissRecord));
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Miss record allocation failed: ") + cudaGetErrorString(cudaResult));
	}

	//Upload packed record
	cudaResult = cudaMemcpy(dev_missRecord, &missRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Miss record upload failed: ") + cudaGetErrorString(cudaResult));
	}

	//Describe the array of miss records: address, spacing, and count
	sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(dev_missRecord);
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = 1;

	//////////////////////////////////
	// Hit record stuff
	//////////////////////////////////
	HitgroupRecord hitgroupRecord = {};
	optixResult = optixSbtRecordPackHeader(hitgroupProgramGroup, &hitgroupRecord);
	checkOptix(optixResult, "Hitgroup record packing failed");

	//Allocate storage for one record on the GPU
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_hitgroupRecord), sizeof(HitgroupRecord));
	checkCuda(cudaResult, "Hitgroup record allocation failed");

	//Copy the packed header into GPU memory
	cudaResult = cudaMemcpy(dev_hitgroupRecord, &hitgroupRecord, sizeof(HitgroupRecord), cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Hitgroup record upload failed");

	//Describe the GPU record array to Optix
	sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(dev_hitgroupRecord);
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = 1;

	//////////////////////////////////
	// Single Triangle Test
	//////////////////////////////////
	const float3 vertices[] = {
	make_float3(-1.0f, -1.0f, 0.0f),
	make_float3(1.0f, -1.0f, 0.0f),
	make_float3(0.0f,  1.0f, 0.0f)
	};

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_testVerticies), sizeof(vertices));
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Triangle vertex allocation failed: ") +cudaGetErrorString(cudaResult));
	}
	cudaResult = cudaMemcpy(dev_testVerticies, vertices, sizeof(vertices), cudaMemcpyHostToDevice);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Triangle vertex upload failed: ") +cudaGetErrorString(cudaResult));
	}

	std::cout << "OptiX test triangle vertices uploaded." << std::endl;

	//////////////////////////////////
	// Describe the triangle and query the GAS(Geometry Acceleration Structure) memory requirements
	//////////////////////////////////

	// It's like glVertexAttribPointer
	// Express GPU pointer using the address type expected by OptiX
	CUdeviceptr vertexBuffer = reinterpret_cast<CUdeviceptr>(dev_testVerticies);
	
	// One geometry-flags entry for one future hitgroup SBT record
	// No need for a any-hit program for the test triangle
	unsigned int triangleFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

	OptixBuildInput triangleInput = {}; //CPU side description of geometry used to build an acceleration structure, just a description, does not hold the vertex data
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; //tells optix that geometry consist sof triangles

	// Each vertex contains 3 floats: x,y,z
	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangleInput.triangleArray.numVertices = 3;

	// Optix expects a CPU array of GPU buffer addresses
	// With no motion blur, one address is enough
	// CPU side OptiX API is reading the address from the CPU memory, GPU doesn't repeatedly fetch that address from the CPU for every ray
	triangleInput.triangleArray.vertexBuffers = &vertexBuffer; //OptiX reads a GPU address stored in CPU memory, then uses the address to locate the vertices

	//No index buffer - every consecutive group of three vertices forms a triangle
	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;

	// All geometry in this input uses one hitgroup SBT record - hitgroup record: a SBT record that selects the programs that handle ray intersections with taht geometry
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.flags = triangleFlags;

	//////////////////////////////////
	// Set build options and ask for the required sizes
	//////////////////////////////////

	//Build a new acceleration structure using default build flags
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	// Optix fills this with the required allocation sizes
	OptixAccelBufferSizes gasBufferSizes = {};

	optixResult = optixAccelComputeMemoryUsage(
		optixContext,
		&accelOptions,
		&triangleInput,
		1, // number of build inputs
		&gasBufferSizes
	);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX GAS memory query failed: ") +optixGetErrorString(optixResult));
	}

	// Scratch space used whiel building the GAS
	std::cout << "GAS temporary memory: " << gasBufferSizes.tempSizeInBytes << " bytes." << std::endl;
	// Storage for the completed GAS, retained while tracing
	std::cout << "GAS output memory: " << gasBufferSizes.outputSizeInBytes << " bytes." << std::endl;

	//////////////////////////////////
	// Create GAS using queried size info from above
	//////////////////////////////////

	// Allocate buffers
	// Allocate the temporary construction workspace
	cudaResult = cudaMalloc(&dev_gasTempBuffer, gasBufferSizes.tempSizeInBytes); //gasBufferSizes was populated from the stage above
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("GAS temporary allocation failed: ") +cudaGetErrorString(cudaResult));
	}

	// Allocate storage for completed GAS
	cudaResult = cudaMalloc(&dev_gasOutputBuffer, gasBufferSizes.outputSizeInBytes);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("GAS output allocation failed: ") +cudaGetErrorString(cudaResult));
	}

	//Build the GAS
	//Build using the same optiosn and triangle description as the size query
	optixResult = optixAccelBuild(
		optixContext,
		nullptr, // use the default CUDA stream
		&accelOptions,
		&triangleInput,
		1, //one build input
		reinterpret_cast<CUdeviceptr>(dev_gasTempBuffer),
		gasBufferSizes.tempSizeInBytes,
		reinterpret_cast<CUdeviceptr>(dev_gasOutputBuffer),
		gasBufferSizes.outputSizeInBytes, 
		&gasHandle, // OptiX writes the resulting handle here
		nullptr, // No optional post-build properties requested
		0 // Number of the properties
	);

if (optixResult != OPTIX_SUCCESS) {
	throw std::runtime_error(std::string("OptiX GAS build failed: ") + optixGetErrorString(optixResult));
}

// Wait for the GPU build to finish before releasing its workspace
cudaResult = cudaDeviceSynchronize();
checkCuda(cudaResult, "GAS build synchronization failed");

// Construction is complete, so the temporary workspace is no longer needed
cudaResult = cudaFree(dev_gasTempBuffer);
if (cudaResult != cudaSuccess) {
	throw std::runtime_error(std::string("GAS temporary buffer cleanup failed: ") + cudaGetErrorString(cudaResult));
}
dev_gasTempBuffer = nullptr;

std::cout << "Optix test triangle GAS built." << std::endl;

//////////////////////////////////
// Uplaod parameters after building the GAS
//////////////////////////////////

//Prepare the parameter values on the CPU
LaunchParams launchParams = {};
launchParams.gasHandle = gasHandle;

//Allocate a GPU buffer for those values
cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_launchParams), sizeof(LaunchParams));
checkCuda(cudaResult, "Launch parameter allocation failed");

//Upload the parameters. This copies the handle, not the GAS itself
cudaResult = cudaMemcpy(dev_launchParams, &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice);
checkCuda(cudaResult, "Launch parameter upload failed");

//Compare it with the value printed by raygen
std::cout << "CPU GAS handle: " << static_cast<unsigned long long>(gasHandle) << std::endl;

//////////////////////////////////
// Launch pipeline
//////////////////////////////////
//std::cout << "Before optixLaunch" << std::endl; //DEBUGUGUUGGUGUGG

optixResult = optixLaunch(
	optixPipeline,
	nullptr, //Default CUDA stream
	reinterpret_cast<CUdeviceptr>(dev_launchParams), //No GPU launch-parameter buffer (yet)
	sizeof(LaunchParams), //Luanch-parameter buffer size
	&sbt, //CPU description pointing to our GPU SBT record
	1, 1, 1 //Launch dimensions (width, height, depth) - OptiX invokes raygen for each launch index, 1, 1, 1 means exactly one invocation
);
//std::cout << "optixLaunch returned: " << optixGetErrorString(optixResult) << std::endl; //DEBUGUGUGGUGUGUGUGUG
checkOptix(optixResult, "OptiX launch failed");

// Launching is asynch, success above does not mean GPU work finished
// Wait for completion, detect execution errors, and flush GPU printf output
//std::cout << "Before synchronization" << std::endl;//DEBUGUGUGUGUGUGUG
cudaResult = cudaDeviceSynchronize();
//std::cout << "Synchronization returned: "<< cudaGetErrorString(cudaResult) << std::endl; //DEBGUUGUGGUGUGUGU
checkCuda(cudaResult, "OptiX GPU execution failed");

std::cout << "OptiX test launch completed." << std::endl;
}

void destroyOptixContext() {
	// The test launch as already snychronized before cleanup.
	if (dev_launchParams != nullptr) {
		cudaError_t result = cudaFree(dev_launchParams);
		checkCuda(result, "Launch parameter cleanup failed");
		dev_launchParams = nullptr;
	}

	// Release gas tempbuffer
	if (dev_gasTempBuffer != nullptr) {
		cudaError_t result = cudaFree(dev_gasTempBuffer);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("GAS temporary buffer cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_gasTempBuffer = nullptr;
	}

	// All GPU work using the GAS must finish before the allocation is freed, our current test already synchronizes before cleanup
	if (dev_gasOutputBuffer != nullptr) {
		cudaError_t result = cudaFree(dev_gasOutputBuffer);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("GAS output buffer cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_gasOutputBuffer = nullptr;
		gasHandle = 0;
	}

	// Release test triangle GPU vertex allocation
	if (dev_testVerticies != nullptr) {
		cudaError_t result = cudaFree(dev_testVerticies);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("Triangle vertex cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_testVerticies = nullptr;
	}

	// Release hit record
	if (dev_hitgroupRecord != nullptr){
		cudaError_t result = cudaFree(dev_hitgroupRecord);
		checkCuda(result, "Hitgroup record cleanup failed");
		dev_hitgroupRecord = nullptr;
		sbt.hitgroupRecordBase = 0;
		sbt.hitgroupRecordStrideInBytes = 0;
		sbt.hitgroupRecordCount = 0;
	}

	// Release miss record
	if (dev_missRecord != nullptr) {
		cudaError_t result = cudaFree(dev_missRecord);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("Miss record cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_missRecord = nullptr;
		sbt.missRecordBase = 0;
		sbt.missRecordStrideInBytes = 0;
		sbt.missRecordCount = 0;
	}

	// Release the GPU memory allcoated for the raygen record
	if (dev_raygenRecord != nullptr) {
		cudaError_t result = cudaFree(dev_raygenRecord);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("Raygen record cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_raygenRecord = nullptr;
		sbt.raygenRecord = 0;
	}

	// Release the pipeline before cleaning up the other OptiX object
	if (optixPipeline != nullptr) {
		OptixResult result = optixPipelineDestroy(optixPipeline);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("OptiX pipeline destruction failed: ") + optixGetErrorString(result));
		}
		optixPipeline = nullptr;
	}

	// Release hit program group
	if (hitgroupProgramGroup != nullptr) {
		OptixResult result = optixProgramGroupDestroy(hitgroupProgramGroup);
		checkOptix(result, "Hitgroup program group destruction failed");
		hitgroupProgramGroup = nullptr;
	}

	//Release miss program group
	if (missProgramGroup != nullptr) {
		OptixResult result = optixProgramGroupDestroy(missProgramGroup);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("Miss program group destruction failed: ") + optixGetErrorString(result));
		}
		missProgramGroup = nullptr;
	}
	
	//Release the program group before releasing its module
	if (raygenProgramGroup != nullptr) {
		OptixResult result = optixProgramGroupDestroy(raygenProgramGroup);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("OptiX program group destruction failed: ") + optixGetErrorString(result));
		}
		raygenProgramGroup = nullptr;
	}

	//Release optix module
	if (optixModule != nullptr) {
		OptixResult result = optixModuleDestroy(optixModule);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("OptiX module destruction failed: ") + optixGetErrorString(result));
		}
		optixModule = nullptr;
	}

	// Nothing to destroy if no context data
	if (optixContext == nullptr) {
		return;
	}
	// Destroy the OptiX context using its handle
	OptixResult optixResult = optixDeviceContextDestroy(optixContext);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX context destruction failed: ") + optixGetErrorString(optixResult));
	}
	// Old handle is no longer valid, clear to make repeated cleanup calls harmless
	optixContext = nullptr;
}