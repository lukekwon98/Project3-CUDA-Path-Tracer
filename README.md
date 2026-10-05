CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)
  

## Readme Outline

- Optix Setup
- Cuda Flowchart x1 + Optix Macro Flowcharts x2 + More specific Optix Flowcharts
  - Performance Comparison
  - Purely Cuda (+ Nsight Systems)
  - Purely Cuda + Stream Compaction (+ Nsight Systems)
  - Optix kernel only for intersection + stream compaction + Nsight Systems
  - Optix kernel only for intersection + Nsight Systems
  - Optix kernel for intersection + shading + Nsight Systems
  - Optix kernel for raygen + intersection + Nsight Systems
  - __inline__ comparison
- Visual Features
  - Refraction, Fresnel, Reflection
  - Physically based materials (Microfacets, Metallic)
  - glTF loading
  - Texture Mapping
  - Environment mapping + MIS (Environment pdf)
  - Metallic Roughness Map
  - Normal Map
  - DOF
  - Reinhard + Gamma



