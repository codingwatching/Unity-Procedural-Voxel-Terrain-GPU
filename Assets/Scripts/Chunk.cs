using System;
using System.Collections;
using OptIn.Voxel;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshCollider))]
public partial class Chunk : MonoBehaviour
{
    TerrainGenerator generator;
    Vector3Int chunkPosition;
    Vector3Int logicalChunkSize;
    Vector3Int paddedChunkSize;

    bool initialized;
    bool dirty;
    bool argent;
    Voxel[] voxels;
    Coroutine meshUpdator;

    // Mesh
    Mesh mesh;
    MeshFilter meshFilter;
    MeshRenderer meshRenderer;
    MeshCollider meshCollider;

    public event Func<bool> CanUpdate;
    GPUVoxelData voxelData;
    VoxelMeshBuilder.NativeMeshData meshData;

    // 缓存的顶点布局参数（不会每次更新时改变）
    VertexAttributeDescriptor[] cachedVertexAttributes;

    public bool Dirty => dirty;
    public bool Updating => meshUpdator != null;
    public bool Initialized => initialized;
    public Voxel[] Voxels => voxels;

    void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();
        meshCollider = GetComponent<MeshCollider>();
        mesh = new Mesh { indexFormat = IndexFormat.UInt32 };
        CanUpdate = () => true;
    }

    void OnDestroy()
    {
        voxelData?.Dispose();
        meshData?.jobHandle.Complete();
        meshData?.Dispose();

        if (mesh != null)
        {
            Destroy(mesh);
            mesh = null;
        }
    }

    void Start()
    {
        meshFilter.mesh = mesh;
    }

    public void Init(Vector3Int position, TerrainGenerator parent)
    {
        chunkPosition = position;
        generator = parent;

        meshRenderer.material = generator.ChunkMaterial;
        logicalChunkSize = generator.ChunkSize;
        paddedChunkSize = logicalChunkSize + Vector3Int.one * 2;
        transform.position = VoxelUtil.ChunkToWorld(chunkPosition, logicalChunkSize);

        cachedVertexAttributes = new VertexAttributeDescriptor[]
        {
            new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3),
            new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3),
            new VertexAttributeDescriptor(VertexAttribute.TexCoord0, VertexAttributeFormat.Float32, 4),
        };

        StartCoroutine(nameof(InitUpdator));
    }

    IEnumerator InitUpdator()
    {
        int numVoxels = paddedChunkSize.x * paddedChunkSize.y * paddedChunkSize.z;
        voxels = new Voxel[numVoxels];
        voxelData = new GPUVoxelData(VoxelUtil.ToInt3(paddedChunkSize));
        // 注意：此处的Compute Shader需要更新以匹配新的Voxel结构体布局才能正确生成密度数据。
        yield return voxelData.Generate(voxels, VoxelUtil.ToInt3(chunkPosition), VoxelUtil.ToInt3(paddedChunkSize), generator.voxelComputeShader);
        dirty = true;
        initialized = true;
    }

    void Update()
    {
        if (!initialized)
            return;

        if (Updating)
            return;

        if (!dirty)
            return;

        if (CanUpdate == null || !CanUpdate())
            return;

        meshUpdator = StartCoroutine(nameof(UpdateMesh));
    }

    IEnumerator UpdateMesh()
    {
        if (Updating)
            yield break;

        if (!generator.CanUpdate)
            yield break;

        generator.UpdatingChunks++;

        meshData?.Dispose();
        meshData = new VoxelMeshBuilder.NativeMeshData(VoxelUtil.ToInt3(paddedChunkSize));
        yield return meshData.ScheduleMeshingJob(voxels, VoxelUtil.ToInt3(paddedChunkSize), argent);

        meshData.GetMeshInformation(out int vertexCount, out int indexCount);

        mesh.Clear();
        if (vertexCount > 0 && indexCount > 0)
        {
            mesh.SetVertexBufferParams(vertexCount, cachedVertexAttributes);
            mesh.SetIndexBufferParams(indexCount, IndexFormat.UInt32);
            // 直接上传更新的 GPUVertex 数据（实际生成的顶点数可能小于最大缓存数）
            mesh.SetVertexBufferData(meshData.nativeVertices, 0, 0, vertexCount, 0, MeshUpdateFlags.DontRecalculateBounds);

            // 上传索引数据
            mesh.SetIndexBufferData(meshData.nativeIndices, 0, 0, indexCount, MeshUpdateFlags.DontRecalculateBounds);

            // 更新子网格描述，告知实际使用的索引数量
            SubMeshDescriptor subMeshDescriptor = new SubMeshDescriptor(0, indexCount, MeshTopology.Triangles);
            mesh.SetSubMesh(0, subMeshDescriptor, MeshUpdateFlags.DontRecalculateBounds);

            // 仅重新计算包围盒，避免重复计算法线
            mesh.RecalculateBounds();

            if (argent)
                SetSharedMesh(mesh);
            else
                VoxelColliderBuilder.Instance.Enqueue(this, mesh);
        }

        meshData.Dispose();
        dirty = false;
        argent = false;
        gameObject.layer = LayerMask.NameToLayer("Voxel");
        meshUpdator = null;
        generator.UpdatingChunks--;
    }

    public void SetSharedMesh(Mesh bakedMesh)
    {
        meshCollider.sharedMesh = bakedMesh;
    }

    public bool GetVoxel(Vector3Int gridPosition, out Voxel voxel)
    {
        if (!initialized)
        {
            voxel = Voxel.Empty;
            return false;
        }

        Vector3Int arrayIndex = gridPosition + Vector3Int.one;

        if (!VoxelUtil.BoundaryCheck(arrayIndex, paddedChunkSize))
        {
            voxel = Voxel.Empty;
            return false;
        }

        voxel = voxels[VoxelUtil.To1DIndex(arrayIndex, paddedChunkSize)];
        return true;
    }

    public bool SetVoxel(Vector3Int gridPosition, Voxel voxel)
    {
        if (!initialized)
        {
            return false;
        }

        Vector3Int arrayIndex = gridPosition + Vector3Int.one;

        if (!VoxelUtil.BoundaryCheck(arrayIndex, paddedChunkSize))
        {
            return false;
        }

        voxels[VoxelUtil.To1DIndex(arrayIndex, paddedChunkSize)] = voxel;
        dirty = true;
        argent = true;
        return true;
    }

    public void NeighborChunkIsChanged()
    {
        dirty = true;
        argent = true;
    }

    void OnDrawGizmos()
    {
        if (logicalChunkSize == Vector3Int.zero) return;

        if (!initialized)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(transform.position + (Vector3)logicalChunkSize / 2f, logicalChunkSize);
        }
        else if (initialized && dirty)
        {
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(transform.position + (Vector3)logicalChunkSize / 2f, logicalChunkSize);
        }
    }
}