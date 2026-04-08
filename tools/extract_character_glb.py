import argparse
import base64
import json
import math
import pathlib
import struct


WHITE_PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=="
)

COMPONENT_FORMATS = {
    5121: ("B", 1),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_glb")
    parser.add_argument("output_mesh")
    parser.add_argument("output_texture")
    return parser.parse_args()


def read_glb(path: pathlib.Path):
    data = path.read_bytes()
    if len(data) < 20:
        raise RuntimeError("GLB file is too small.")

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise RuntimeError("Invalid GLB magic.")
    if version != 2:
        raise RuntimeError(f"Unsupported GLB version: {version}")
    if length != len(data):
        raise RuntimeError("GLB length header does not match file size.")

    offset = 12
    json_chunk = None
    bin_chunk = b""

    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset:offset + chunk_length]
        offset += chunk_length

        if chunk_type == 0x4E4F534A:
            json_chunk = chunk_data
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk_data

    if json_chunk is None:
        raise RuntimeError("GLB JSON chunk not found.")

    return json.loads(json_chunk.decode("utf-8")), bin_chunk


def identity_matrix():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for row in range(4):
        for col in range(4):
            out[row][col] = (
                a[row][0] * b[0][col] +
                a[row][1] * b[1][col] +
                a[row][2] * b[2][col] +
                a[row][3] * b[3][col]
            )
    return out


def transform_point(matrix, point):
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def matrix_from_gltf(raw_values):
    return [
        [raw_values[0], raw_values[4], raw_values[8], raw_values[12]],
        [raw_values[1], raw_values[5], raw_values[9], raw_values[13]],
        [raw_values[2], raw_values[6], raw_values[10], raw_values[14]],
        [raw_values[3], raw_values[7], raw_values[11], raw_values[15]],
    ]


def matrix_from_trs(node):
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])

    x, y, z, w = rotation
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation_matrix = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    scale_matrix = [
        [scale[0], 0.0, 0.0, 0.0],
        [0.0, scale[1], 0.0, 0.0],
        [0.0, 0.0, scale[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    translation_matrix = identity_matrix()
    translation_matrix[0][3] = translation[0]
    translation_matrix[1][3] = translation[1]
    translation_matrix[2][3] = translation[2]

    return mat_mul(translation_matrix, mat_mul(rotation_matrix, scale_matrix))


def get_node_matrix(node):
    if "matrix" in node:
        return matrix_from_gltf(node["matrix"])
    return matrix_from_trs(node)


def read_accessor(document, binary_chunk, accessor_index):
    accessor = document["accessors"][accessor_index]
    if "bufferView" not in accessor:
        raise RuntimeError("Sparse accessors are not supported.")

    buffer_view = document["bufferViews"][accessor["bufferView"]]
    component_type = accessor["componentType"]
    accessor_type = accessor["type"]
    count = accessor["count"]

    if component_type not in COMPONENT_FORMATS:
        raise RuntimeError(f"Unsupported accessor component type: {component_type}")
    if accessor_type not in TYPE_COMPONENTS:
        raise RuntimeError(f"Unsupported accessor type: {accessor_type}")

    fmt, component_size = COMPONENT_FORMATS[component_type]
    component_count = TYPE_COMPONENTS[accessor_type]
    packed_size = component_size * component_count
    stride = buffer_view.get("byteStride", packed_size)

    buffer_offset = buffer_view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    start = buffer_offset + accessor_offset

    values = []
    for element_index in range(count):
        element_offset = start + element_index * stride
        element = struct.unpack_from("<" + fmt * component_count, binary_chunk, element_offset)
        if component_count == 1:
            values.append(element[0])
        else:
            values.append(tuple(element))

    return values


def resolve_image_bytes(document, binary_chunk, glb_path, image_index):
    image = document["images"][image_index]

    if "bufferView" in image:
        buffer_view = document["bufferViews"][image["bufferView"]]
        offset = buffer_view.get("byteOffset", 0)
        length = buffer_view["byteLength"]
        return binary_chunk[offset:offset + length]

    uri = image.get("uri")
    if not uri:
        return WHITE_PNG_BYTES

    if uri.startswith("data:"):
        _, encoded = uri.split(",", 1)
        return base64.b64decode(encoded)

    return (glb_path.parent / uri).read_bytes()


def find_base_color_image(document, primitive, binary_chunk, glb_path):
    material_index = primitive.get("material")
    if material_index is not None:
        material = document["materials"][material_index]
        pbr = material.get("pbrMetallicRoughness", {})
        base_color_texture = pbr.get("baseColorTexture")
        if base_color_texture is not None:
            texture_index = base_color_texture["index"]
            texture = document["textures"][texture_index]
            source_index = texture["source"]
            return resolve_image_bytes(document, binary_chunk, glb_path, source_index)

    if document.get("images"):
        return resolve_image_bytes(document, binary_chunk, glb_path, 0)

    return WHITE_PNG_BYTES


def load_scene_primitives(document, binary_chunk):
    scene_index = document.get("scene", 0)
    scene = document["scenes"][scene_index]
    nodes = document["nodes"]
    meshes = document["meshes"]

    collected = []

    def walk(node_index, parent_matrix):
        node = nodes[node_index]
        world_matrix = mat_mul(parent_matrix, get_node_matrix(node))

        if "mesh" in node:
            mesh = meshes[node["mesh"]]
            for primitive in mesh.get("primitives", []):
                collected.append((primitive, world_matrix))

        for child_index in node.get("children", []):
            walk(child_index, world_matrix)

    for root_node in scene.get("nodes", []):
        walk(root_node, identity_matrix())

    return collected


def extract_mesh(document, binary_chunk, glb_path):
    vertices = []
    indices = []
    texture_bytes = None

    primitives = load_scene_primitives(document, binary_chunk)
    if not primitives:
        raise RuntimeError("No mesh primitives found in Character.glb.")

    for primitive, world_matrix in primitives:
        if primitive.get("mode", 4) != 4:
            continue

        attributes = primitive["attributes"]
        if "POSITION" not in attributes:
            continue

        positions = read_accessor(document, binary_chunk, attributes["POSITION"])
        texcoords = read_accessor(document, binary_chunk, attributes["TEXCOORD_0"]) if "TEXCOORD_0" in attributes else None
        primitive_indices = read_accessor(document, binary_chunk, primitive["indices"]) if "indices" in primitive else list(range(len(positions)))

        if texture_bytes is None:
            texture_bytes = find_base_color_image(document, primitive, binary_chunk, glb_path)

        base_vertex = len(vertices)
        for vertex_index, position in enumerate(positions):
            px, py, pz = transform_point(world_matrix, position)
            if texcoords is not None:
                u, v = texcoords[vertex_index]
            else:
                u, v = 0.0, 0.0
            vertices.append((px, py, pz, float(u), float(v)))

        for index in primitive_indices:
            indices.append(base_vertex + int(index))

    if not vertices or not indices:
        raise RuntimeError("Character.glb did not contain a triangle mesh with positions.")

    if texture_bytes is None:
        texture_bytes = WHITE_PNG_BYTES

    return vertices, indices, texture_bytes


def write_mesh(path: pathlib.Path, vertices, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"PMSH", len(vertices), len(indices)))
        for vertex in vertices:
            f.write(struct.pack("<5f", *vertex))
        for index in indices:
            f.write(struct.pack("<I", index))


def main():
    args = parse_args()
    input_glb = pathlib.Path(args.input_glb)
    output_mesh = pathlib.Path(args.output_mesh)
    output_texture = pathlib.Path(args.output_texture)

    document, binary_chunk = read_glb(input_glb)
    vertices, indices, texture_bytes = extract_mesh(document, binary_chunk, input_glb)

    write_mesh(output_mesh, vertices, indices)
    output_texture.parent.mkdir(parents=True, exist_ok=True)
    output_texture.write_bytes(texture_bytes)


if __name__ == "__main__":
    main()
