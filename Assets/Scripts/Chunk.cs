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
    Vector3Int chunkSize;

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

    // 顶点属性缓存
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
        chunkSize = generator.ChunkSize;

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
        int numVoxels = chunkSize.x * chunkSize.y * chunkSize.z;
        voxels = new Voxel[numVoxels];
        voxelData = new GPUVoxelData(VoxelUtil.ToInt3(chunkSize));
        yield return voxelData.Generate(voxels, VoxelUtil.ToInt3(chunkPosition), VoxelUtil.ToInt3(chunkSize), generator.voxelComputeShader, generator.SimplifyingMethod);
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
        meshData = new VoxelMeshBuilder.NativeMeshData(VoxelUtil.ToInt3(chunkSize));
        yield return meshData.ScheduleMeshingJob(voxels, VoxelUtil.ToInt3(chunkSize), generator.SimplifyingMethod, argent);

        meshData.GetMeshInformation(out int vertexCount, out int indexCount);

        if (vertexCount > 0 && indexCount > 0)
        {
            mesh.Clear();

            mesh.SetVertexBufferParams(vertexCount, cachedVertexAttributes);
            mesh.SetIndexBufferParams(indexCount, IndexFormat.UInt32);
            mesh.SetVertexBufferData(meshData.nativeVertices, 0, 0, vertexCount, 0, MeshUpdateFlags.DontRecalculateBounds);
            mesh.SetIndexBufferData(meshData.nativeIndices, 0, 0, indexCount, MeshUpdateFlags.DontRecalculateBounds);

            SubMeshDescriptor subMeshDescriptor = new SubMeshDescriptor(0, indexCount, MeshTopology.Triangles);
            mesh.SetSubMesh(0, subMeshDescriptor, MeshUpdateFlags.DontRecalculateBounds);

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

    // 修改后的 GetVoxel：支持超出本区块时查询邻区块的体素数据
    public bool GetVoxel(Vector3Int gridPosition, out Voxel voxel)
    {
        if (!initialized)
        {
            voxel = Voxel.Empty;
            return false;
        }
        if (VoxelUtil.BoundaryCheck(gridPosition, chunkSize))
        {
            voxel = voxels[VoxelUtil.To1DIndex(gridPosition, chunkSize)];
            return true;
        }
        else
        {
            // 将超出边界的网格坐标转换为世界坐标，再查询对应的区块
            Vector3 worldPos = VoxelUtil.GridToWorld(gridPosition, chunkPosition, chunkSize);
            if (TerrainGenerator.Instance.GetChunk(worldPos, out Chunk neighborChunk))
            {
                Vector3Int neighborGridPos = VoxelUtil.WorldToGrid(worldPos, VoxelUtil.WorldToChunk(worldPos, chunkSize), chunkSize);
                return neighborChunk.GetVoxel(neighborGridPos, out voxel);
            }
            voxel = Voxel.Empty;
            return false;
        }
    }

    // 修改后的 SetVoxel：支持超出本区块时更新邻区块的体素数据
    public bool SetVoxel(Vector3Int gridPosition, ushort type)
    {
        if (!initialized)
            return false;
        if (VoxelUtil.BoundaryCheck(gridPosition, chunkSize))
        {
            int index = VoxelUtil.To1DIndex(gridPosition, chunkSize);
            Voxel voxel = voxels[index];
            voxel.texId = type;
            voxel.shapeId = 0;
            // 当 type==1 时认为为固体，density 取正数；否则为空气，density 取负数
            voxel.Density = (type == 1) ? 1f : -1f;
            voxels[index] = voxel;
            dirty = true;
            argent = true;
            return true;
        }
        else
        {
            Vector3 worldPos = VoxelUtil.GridToWorld(gridPosition, chunkPosition, chunkSize);
            if (TerrainGenerator.Instance.GetChunk(worldPos, out Chunk neighborChunk))
            {
                Vector3Int neighborGridPos = VoxelUtil.WorldToGrid(worldPos, VoxelUtil.WorldToChunk(worldPos, chunkSize), chunkSize);
                return neighborChunk.SetVoxel(neighborGridPos, type);
            }
            return false;
        }
    }

    public void NeighborChunkIsChanged()
    {
        dirty = true;
        argent = true;
    }

    void OnDrawGizmos()
    {
        if (!initialized)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(transform.position + new Vector3(chunkSize.x / 2f, chunkSize.y / 2f, chunkSize.z / 2f), chunkSize);
        }
        else if (initialized && dirty)
        {
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(transform.position + new Vector3(chunkSize.x / 2f, chunkSize.y / 2f, chunkSize.z / 2f), chunkSize);
        }
    }
}
