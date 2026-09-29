// Compile TinyGLTF's implementation in this file only
#define TINYGLTF_IMPLEMENTATION

// Geometry only loading, no textures
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "tiny_gltf.h"
#include "gltfLoader.h"
#include <iostream>
#include <cstring> //for std::memcpy
#include <cstdint> //std::uint16_t
#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include <utility>

struct MeshInstance {
	int meshIndex;
	glm::mat4 worldTransform;
};

glm::mat4 getLocalTransform(const tinygltf::Node& node) {
	if (!node.matrix.empty()) {
		if (node.matrix.size() != 16) {
			throw std::runtime_error("Invalid node matrix size");
		}

		return glm::mat4(glm::make_mat4(node.matrix.data()));
	}

	glm::vec3 translation(0.0f);
	glm::vec3 scale(1.0f);
	glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

	if (!node.translation.empty()) {
		translation = glm::vec3(node.translation.at(0), node.translation.at(1), node.translation.at(2));
	}

	if (!node.scale.empty()) {
		scale = glm::vec3(node.scale.at(0), node.scale.at(1), node.scale.at(2));
	}

	//gltf: xyzw, glm: wxyz
	if (!node.rotation.empty()) {
		rotation = glm::quat(static_cast<float>(node.rotation.at(3)), static_cast<float>(node.rotation.at(0)), static_cast<float>(node.rotation.at(1)), static_cast<float>(node.rotation.at(2)));
	}

	return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

void traverseMesh(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentWorld, std::vector <MeshInstance>& instances) {
	const tinygltf::Node& node = model.nodes.at(nodeIndex);

	const glm::mat4 world = parentWorld * getLocalTransform(node);

	if (node.mesh >= 0) {
		instances.push_back({ node.mesh, world });
	}

	for (int childIndex : node.children) {
		traverseMesh(model, childIndex, world, instances);
	}
}

bool loadGltf(const std::string& filename, std::vector<MeshData>& output) {
	std::vector<MeshData> loadedMeshes;

	tinygltf::TinyGLTF loader; //performs parcing
	tinygltf::Model model; //owns the loaded CPU data
	std::string error;
	std::string warning;
	//a gltf primitive is a drawable mesh section
	//a gltf primitive is a group of geometry within a mesh that uses one material

	bool loaded = loader.LoadASCIIFromFile(&model, &error, &warning, filename);

	if (!warning.empty()) {
		std::cerr << "glTF warning: " << warning << '\n';
	}

	if (!error.empty()) {
		std::cerr << "glTF error: " << error << '\n';
	}

	if (!loaded) {
		std::cerr << "Failed to load: " << filename << '\n';
		return false;
	}

	std::cout << "Loaded: " << filename << '\n'
		<< "Scenes: " << model.scenes.size() << '\n'
		<< "Nodes: " << model.nodes.size() << '\n'
		<< "Meshes: " << model.meshes.size() << '\n'
		<< "Buffers: " << model.buffers.size() << '\n';

	//////////////////
	// Scene check
	//////////////////
	if (model.scenes.empty()) {
		std::cerr << "glTF has no scenes\n";
		return false;
	}

	// Use the declared default scene, ro choose scene 0 if none is declared
	const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
	const tinygltf::Scene& gltfScene = model.scenes.at(sceneIndex);
	std::vector<MeshInstance> instances;

	for (int rootNode : gltfScene.nodes) {
		traverseMesh(model, rootNode, glm::mat4(1.0f), instances);
	}

	//////////////////
	// Model Loop
	//////////////////
	for (const MeshInstance& instance: instances) {
		const tinygltf::Mesh& mesh = model.meshes.at(instance.meshIndex);

		std::cout << "Mesh " << instance.meshIndex << ": " << mesh.primitives.size() << " primitive(s)\n";

		//////////////////
		// Mesh Loop
		//////////////////
		for (size_t p = 0; p < mesh.primitives.size(); p++) {
			const tinygltf::Primitive& primitive = mesh.primitives[p];

			auto positionIt = primitive.attributes.find("POSITION");
			if (positionIt == primitive.attributes.end()) {
				std::cerr << "Primitive has no POSITION attribute\n";
				return false;
			}

			//Accessors describe how to interpret data (element count, type, and location in a buffer), it follows the gltf format
			const tinygltf::Accessor& positions = model.accessors.at(positionIt->second);

			// Supports ordinary FLOAT VEC3 positions
			if (positions.type != TINYGLTF_TYPE_VEC3 ||
				positions.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
				positions.sparse.isSparse ||
				positions.bufferView < 0) {
				std::cerr << "Unsupported position accessor\n";
				return false;
			}

			// Accessor -> buffer view -> buffer
			const tinygltf::BufferView& view = model.bufferViews.at(positions.bufferView); //A specified region within a buffer
			const tinygltf::Buffer& buffer = model.buffers.at(view.buffer); //Raw bytes loaded into memory

			// Each position contains 3 floats
			const size_t elementBytes = 3 * sizeof(float);
			// A missing byteStride means elements are tightly packed
			const size_t stride = view.byteStride != 0 ? view.byteStride : elementBytes;

			if (view.byteOffset > buffer.data.size() ||
				view.byteLength > buffer.data.size() - view.byteOffset ||
				positions.byteOffset > view.byteLength ||
				stride < elementBytes) {
				std::cerr << "Invalid position buffer layout\n";
				return false;
			}

			const size_t available = view.byteLength - positions.byteOffset;

			if (positions.count > 0 && (available < elementBytes || positions.count - 1 > (available - elementBytes) / stride)){
				std::cerr << "Position data exceeds buffer view\n";
				return false;
			}

			const size_t start = view.byteOffset + positions.byteOffset;

			//////////////////
			// Positions Loop
			//////////////////
			std::vector<glm::vec3> meshPositions;
			meshPositions.reserve(positions.count);

			for (size_t v = 0; v < positions.count; v++) {
				float xyz[3];

				//copy into xyz, buffer.data with offset start + v * stride, with increments of 1 to v
				std::memcpy(xyz, buffer.data.data() + start + v * stride, sizeof(xyz));

				meshPositions.emplace_back(xyz[0], xyz[1], xyz[2]);

				//std::cout << "Position" << v << ": " << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << "\n";
			}

			for (glm::vec3& position : meshPositions) {
				position = glm::vec3(instance.worldTransform * glm::vec4(position, 1.0f));
			}

			if (!meshPositions.empty()) {
				const glm::vec3 & first = meshPositions.front();

				std::cout << "First world position: " << first.x << ", " << first.y << ", " << first.z << '\n';
			}

			std::cout << " Primitive " << p << '\n'
					  << "    Mode: " << primitive.mode << '\n'
					  << "    Vertices: " << positions.count << '\n'
					  << "    Position type: " << positions.type << '\n'
					  << "    Position component type: " << positions.componentType << '\n';


			std::vector<glm::vec3> meshNormals;

			auto normalIt = primitive.attributes.find("NORMAL");

			if (normalIt != primitive.attributes.end()) {
				const tinygltf::Accessor& normals = model.accessors.at(normalIt->second);
				if (normals.type != TINYGLTF_TYPE_VEC3 ||
					normals.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
					normals.sparse.isSparse ||
					normals.bufferView < 0 ||
					normals.count != positions.count) {
					std::cerr << "Unsupported normal accessor\n";
					return false;
				}

				std::cout << "    Normals: " << normals.count << '\n';

				const tinygltf::BufferView& normalView = model.bufferViews.at(normals.bufferView);
				const tinygltf::Buffer& normalBuffer = model.buffers.at(normalView.buffer);

				const size_t normalElementBytes = 3 * sizeof(float);
				const size_t normalStride = normalView.byteStride != 0 ? normalView.byteStride : normalElementBytes;

				// Validate
				if (normalView.byteOffset > normalBuffer.data.size() ||
					normalView.byteLength > normalBuffer.data.size() - normalView.byteOffset ||
					normals.byteOffset > normalView.byteLength ||
					normalStride < normalElementBytes) {
					std::cerr << "Invalid normal buffer layout\n";
					return false;
				}

				const size_t availableNormalBytes = normalView.byteLength - normals.byteOffset;

				if (normals.count > 0 && availableNormalBytes < normalElementBytes ||
					normals.count - 1 >(availableNormalBytes - normalElementBytes) / normalStride) {
					std::cerr << "Normal data exceeds buffer view\n";
					return false;
				}

				const size_t normalStart = normalView.byteOffset + normals.byteOffset;

				meshNormals.reserve(normals.count);

				for (size_t v = 0; v < normals.count; v++) {
					float xyz[3];

					std::memcpy(xyz, normalBuffer.data.data() + normalStart + v * normalStride, sizeof(xyz));

					meshNormals.push_back(glm::vec3(xyz[0], xyz[1], xyz[2]));
				}

				const glm::mat3 linearTransform = glm::mat3(instance.worldTransform);
				if (glm::determinant(linearTransform) == 0.0f) {
					std::cerr << "Cannot transform normasl with a singular transform\n";
					return false;
				}

				const glm::mat3 normalMatrix = glm::transpose(glm::inverse(linearTransform)); //inverseTranspose undoes rotation, only scale
				
				for (glm::vec3& normal : meshNormals) {
					glm::vec3 worldNormal = normalMatrix * normal;

					if (glm::dot(worldNormal, worldNormal) == 0.0f) {
						std::cerr << "Invalid zero-length normal\n";
						return false;
					}
					normal = glm::normalize(worldNormal);
				}
			}

			std::vector<std::array<std::uint32_t, 3>> meshTriangles;

			if (primitive.indices < 0) {
				std::cerr << "Non-indexed geometry is not supported yet\n";
				return false;
			}

			//////////////////
			// Indices Loop
			//////////////////
			if (primitive.indices >= 0) {
				const tinygltf::Accessor& indices = model.accessors.at(primitive.indices);

				std::cout << "    Indices: " << indices.count << '\n'
						  << "    Index component type: "
						  << indices.componentType << '\n';

				//Indexed triangles with unsigned 16 bit indices
				if (primitive.mode != TINYGLTF_MODE_TRIANGLES ||
					indices.type != TINYGLTF_TYPE_SCALAR ||
					indices.sparse.isSparse ||
					indices.bufferView < 0 ||
					indices.count % 3 != 0) {
					std::cerr << "Unsupported triangle index format\n";
					return false;
				}

				const tinygltf::BufferView& indexView = model.bufferViews.at(indices.bufferView);
				const tinygltf::Buffer& indexBuffer = model.buffers.at(indexView.buffer);

				size_t indexBytes = 0;

				switch (indices.componentType) {
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
					indexBytes = sizeof(std::uint8_t);
					break;
					
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
					indexBytes = sizeof(std::uint16_t);
					break;

				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
					indexBytes = sizeof(std::uint32_t);
					break;

				default:
					std::cerr << "Unsupported index component type\n";
					return false;
				}

				//Index data is tightly packed
				if (indexView.byteStride != 0 ||
					indexView.byteOffset > indexBuffer.data.size() ||
					indexView.byteLength > indexBuffer.data.size() - indexView.byteOffset ||
					indices.byteOffset > indexView.byteLength) {
					std::cerr << "Invalid index buffer layout\n";
					return false;
				}

				const size_t availableIndexBytes = indexView.byteLength - indices.byteOffset;

				if (indices.count > availableIndexBytes / indexBytes) {
					std::cerr << "Index data exceeds buffer view\n";
					return false;
				}

				const size_t indexStart = indexView.byteOffset + indices.byteOffset;

				meshTriangles.reserve(indices.count / 3);

				//GLTF has different types of requirements
				for (size_t t = 0; t < indices.count / 3; ++t) {
					std::uint32_t triangle[3]; //triangle is just a set of 3 shorts from indices
					for (size_t corner = 0; corner < 3; corner++) {
						const unsigned char* source = indexBuffer.data.data() + indexStart + (t * 3 + corner) * indexBytes;

						switch (indices.componentType) {
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
							std::uint8_t value;
							std::memcpy(&value, source, sizeof(value));
							triangle[corner] = value;
							break;
						}

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
							std::uint16_t value;
							std::memcpy(&value, source, sizeof(value));
							triangle[corner] = value;
							break;
						}

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
							std::uint32_t value;
							std::memcpy(&value, source, sizeof(value));
							triangle[corner] = value;
							break;
						}
						}
					}

					if (triangle[0] >= positions.count ||
						triangle[1] >= positions.count ||
						triangle[2] >= positions.count) {
						std::cerr << "Triangle references an invalid vertex\n";
						return false;
					}

					meshTriangles.push_back({ triangle[0], triangle[1], triangle[2] });

					//std::cout << "Triangle " << t << ": " << triangle[0] << ", " << triangle[1] << ", " << triangle[2] << '\n';
				}
			}
			std::cout << "Stored " << meshPositions.size() << " positions and " << meshTriangles.size() << " triangles\n";

			MeshData meshData;
			meshData.positions = std::move(meshPositions);
			meshData.normals = std::move(meshNormals);
			meshData.triangles = std::move(meshTriangles);
			meshData.gltfMaterialIndex = primitive.material;

			if (primitive.material >= 0) {
				if (static_cast<size_t>(primitive.material) >= model.materials.size()) {
					std::cerr << "Invalid material index\n";
					return false;
				}

				//get material property
				const tinygltf::Material& material = model.materials.at(primitive.material);

				//get base color from materials
				const auto& color = material.pbrMetallicRoughness.baseColorFactor;

				if (color.size() != 4) {
					std::cerr << "Invalid base color factor\n";
					return false;
				}

				meshData.baseColorFactor = glm::vec4(static_cast<float>(color[0]), static_cast<float>(color[1]), static_cast<float>(color[2]), static_cast<float>(color[3]));
			}

			std::cout << "Base color: "
				<< meshData.baseColorFactor.r << ", "
				<< meshData.baseColorFactor.g << ", "
				<< meshData.baseColorFactor.b << ", "
				<< meshData.baseColorFactor.a << '\n';

			loadedMeshes.push_back(std::move(meshData));
		}
	}

	output = std::move(loadedMeshes); //lets the destination vectors take ownership of the existing allocations instead of copying every vertex and triangle
	return true;
}