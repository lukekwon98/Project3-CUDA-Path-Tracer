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
  - __forceinline__ comparison (uber kernel & branching) - if I have time
- Visual Features
  - Jitter anti-aliasing
  - Refraction, Fresnel, Reflection
  - Physically based materials (Microfacets, Metallic)
  - glTF loading
  - Texture Mapping
  - Environment mapping + MIS (Environment pdf)
  - Metallic Roughness Map
  - Normal Map
  - DOF
  - Reinhard + Gamma
 - Mention cmakeslists changes as mentioned in instructions

<img width="800" height="800" alt="Reflective2" src="https://github.com/user-attachments/assets/65ed8a8e-f0e7-4b46-8354-657deec5012c" />
<img width="800" height="800" alt="Reflective1" src="https://github.com/user-attachments/assets/d560495f-a8d7-4db4-86e4-30f6ff0ced62" />
<img width="800" height="800" alt="Reflective" src="https://github.com/user-attachments/assets/5aeab3fe-8a2f-4fd4-b424-8e7e98f33d5c" />
<img width="800" height="800" alt="NotFinal" src="https://github.com/user-attachments/assets/7f69a0ba-f6ef-45e6-8ffa-d7ec2b07e257" />
<img width="800" height="800" alt="NormalMap" src="https://github.com/user-attachments/assets/36eb52eb-9690-4619-814b-f2e5253790e4" />
<img width="800" height="800" alt="No MIS" src="https://github.com/user-attachments/assets/4c4cbd31-4290-49ab-8a36-76444e5dcd8e" />
<img width="800" height="800" alt="MetallicRoughnessMap" src="https://github.com/user-attachments/assets/4da282d2-7856-4a8b-b0ac-6037eaf65571" />
<img width="800" height="800" alt="Metallic Roughness" src="https://github.com/user-attachments/assets/b8770a1d-44ca-4cf3-8073-84980035a6f7" />
<img width="800" height="800" alt="Metallic Roughness Gold" src="https://github.com/user-attachments/assets/93ab6e98-6aac-4109-8491-e92f9f487d70" />
<img width="800" height="800" alt="Metallic Roughness Blue" src="https://github.com/user-attachments/assets/c62bac1c-4c05-4cc9-bca6-f9104cac4fbc" />
<img width="580" height="512" alt="JitterX_Cornell" src="https://github.com/user-attachments/assets/05b47519-6486-4388-b0fe-94559746b913" />
<img width="2061" height="1118" alt="JitterX" src="https://github.com/user-attachments/assets/c4ff0c9f-5d0e-430b-84ab-a97fb8eb20b2" />
<img width="560" height="492" alt="JitterO_Cornell" src="https://github.com/user-attachments/assets/b816cc70-a4e2-4987-bec5-603dba39aed3" />
<img width="2075" height="1197" alt="JitterO" src="https://github.com/user-attachments/assets/8ea4ffec-e8c1-4e3d-9395-3d98440544d2" />
<img width="800" height="800" alt="Environment Map MIS" src="https://github.com/user-attachments/assets/fa2c5f14-f4a9-4186-ab7c-dc1be2c8a0a5" />
<img width="1150" height="760" alt="cornell 2026-10-07_11-57-11z 50000samp" src="https://github.com/user-attachments/assets/4c291d1b-6c14-4230-a87c-b7dd20ddd155" />
<img width="800" height="800" alt="AllTexturesCombined" src="https://github.com/user-attachments/assets/43dfa633-abe1-4518-8aec-3ab391c037ed" />
<img width="1150" height="760" alt="6" src="https://github.com/user-attachments/assets/52349970-5202-43f4-8761-e4c857d4fcfb" />
<img width="1150" height="760" alt="5" src="https://github.com/user-attachments/assets/6b081d53-71d7-45c5-96e4-296da682ee11" />
<img width="1150" height="760" alt="4" src="https://github.com/user-attachments/assets/ad3da490-d793-4595-acb9-d72624c7ecaf" />
<img width="1150" height="760" alt="3" src="https://github.com/user-attachments/assets/9e88f119-b532-464f-91b4-1011f2da8805" />
<img width="1000" height="800" alt="2" src="https://github.com/user-attachments/assets/66a21fa1-66bb-477f-b683-75c02681bcb7" />
<img width="800" height="800" alt="1" src="https://github.com/user-attachments/assets/9b3ce840-2d83-4ab5-948b-c7c038f368a6" />
<img width="800" height="800" alt="Refractive2" src="https://github.com/user-attachments/assets/c67598d9-b398-46ae-924d-a5ca8ce4b23e" />
<img width="800" height="800" alt="Refractive1" src="https://github.com/user-attachments/assets/eff32a26-d407-43f0-9589-99c5fed35f13" />
<img width="800" height="800" alt="TestObject2_Suzanne3936" src="https://github.com/user-attachments/assets/77171d03-9835-409a-a178-f7dc240016ff" />
<img width="800" height="800" alt="TestObject1_Box12" src="https://github.com/user-attachments/assets/fc3befeb-e141-429b-98f5-e413fbe92649" />
<img width="800" height="800" alt="RoughnessP2F" src="https://github.com/user-attachments/assets/d84a16ef-6f20-4d32-b7ae-543cf14ec54d" />
<img width="800" height="800" alt="RoughnessP2" src="https://github.com/user-attachments/assets/b1ff3d82-a87a-4da0-a87c-9b1f5e6de6dc" />
<img width="800" height="800" alt="RoughnessHF" src="https://github.com/user-attachments/assets/b6c4d600-f27a-4581-a4f5-693d5f85480b" />
<img width="800" height="800" alt="RoughnessH" src="https://github.com/user-attachments/assets/8e820d09-2ef3-4ecb-a8a8-ea9228ff5dcf" />
<img width="800" height="800" alt="Roughness1F" src="https://github.com/user-attachments/assets/de16eb16-8c9e-4c5c-9518-f79e0bdda7f4" />
<img width="800" height="800" alt="Roughness1" src="https://github.com/user-attachments/assets/289351dd-12e5-47a9-9888-5d6efa31b95e" />
<img width="800" height="800" alt="Roughness0F" src="https://github.com/user-attachments/assets/a45110fd-82f4-4a7f-83b0-68b563afeec6" />
<img width="800" height="800" alt="Roughness0" src="https://github.com/user-attachments/assets/4b088eed-bc23-4c6e-a3d3-9fcbe821778d" />
<img width="800" height="800" alt="TextureMap" src="https://github.com/user-attachments/assets/e27e00d8-9869-4a39-9d99-0227dd082011" />
<img width="800" height="800" alt="TestObject5_OccludedBackground" src="https://github.com/user-attachments/assets/e02a7a57-9da3-40c8-9d59-c5864f8c7c77" />
<img width="800" height="800" alt="TestObject4_Sponza262267" src="https://github.com/user-attachments/assets/34f7b529-89e4-4e50-808f-4b02bb25e5a2" />
<img width="800" height="800" alt="TestObject3_FlightHelmet94722" src="https://github.com/user-attachments/assets/c3f66374-f247-4b0b-8601-b30db7042e9a" />


