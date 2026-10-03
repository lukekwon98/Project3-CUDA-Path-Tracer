#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>

struct MeshData {
	std::vector<glm::vec3> positions; //World space
	std::vector<glm::vec3> normals;
	std::vector<std::array<std::uint32_t, 3>> triangles;

	//Index into the glTF material array, not the renderer's array
	int gltfMaterialIndex = -1;
	//Index into renderer's material array
	int rendererMaterialId = -1;
	float metallicFactor = 1.0f;
	float roughnessFactor = 1.0f;
	glm::vec4 baseColorFactor = glm::vec4(1.0f);
};

bool loadGltf(const std::string& filename, std::vector<MeshData>& output);

//Box.gltf for reference
//{
//    "asset": {
//        "generator": "COLLADA2GLTF",
//            "version" : "2.0"
//    },
//        "scene" : 0,
//        "scenes" : [
//    {
//        "nodes": [
//            0
//        ]
//    }
//        ] ,
//        "nodes": [
//    {
//        "children": [
//            1
//        ] ,
//            "matrix" : [
//                1.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                -1.0,
//                0.0,
//                0.0,
//                1.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                1.0
//            ]
//    },
//        {
//            "mesh": 0
//        }
//        ],
//        "meshes": [
//    {
//        "primitives": [
//        {
//            "attributes": {
//                "NORMAL": 1, //first element of attributes is normal?
//                    "POSITION" : 2 //2nd element of attributes is position?
//            },
//                "indices" : 0,
//                "mode" : 4,
//                "material" : 0
//        }
//        ] ,
//            "name": "Mesh"
//    }
//        ],
//        "accessors": [ //accessor
//    {
//        "bufferView": 0, // accessor.at(0) = indices?
//            "byteOffset" : 0,
//            "componentType" : 5123,
//            "count" : 36,
//            "max" : [
//                23
//            ] ,
//            "min" : [
//                0
//            ] ,
//            "type" : "SCALAR"
//    },
//        {
//            "bufferView": 1, // accessor.at(1) = normals
//            "byteOffset" : 0,
//            "componentType" : 5126,
//            "count" : 24,
//            "max" : [
//                1.0,
//                1.0,
//                1.0
//            ] ,
//            "min" : [
//                -1.0,
//                -1.0,
//                -1.0
//            ] ,
//            "type" : "VEC3"
//        },
//        {
//            "bufferView": 1, // accesor.at(2) = position
//            "byteOffset" : 288,
//            "componentType" : 5126,
//            "count" : 24,
//            "max" : [
//                0.5,
//                0.5,
//                0.5
//            ] ,
//            "min" : [
//                -0.5,
//                -0.5,
//                -0.5
//            ] ,
//            "type" : "VEC3"
//        }
//        ],
//        "materials": [
//    {
//        "pbrMetallicRoughness": {
//            "baseColorFactor": [
//                0.800000011920929,
//                0.0,
//                0.0,
//                1.0
//            ] ,
//                "metallicFactor": 0.0
//        },
//            "name": "Red"
//    }
//        ],
//        "bufferViews": [ //what is this
	//    {
	//			"buffer": 0,
	//            "byteOffset" : 576,
	//            "byteLength" : 72,
	//            "target" : 34963
	//    },
//        {
//            "buffer": 0,
//            "byteOffset" : 0,
//            "byteLength" : 576,
//            "byteStride" : 12,
//            "target" : 34962
//        }
//        ],
//        "buffers": [
//    {
//        "byteLength": 648,
//            "uri" : "Box0.bin"
//    }
//        ]
//}
