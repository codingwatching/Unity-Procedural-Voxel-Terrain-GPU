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

    /// <summary>
    /// 从相邻区块更新此区块的Padding体素数据，以确保网格在边界处正确连接。
    /// </summary>
    private void UpdatePaddingVoxels()
    {
        if (generator == null) return;

        // 遍历26个方向的邻居
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    if (dx == 0 && dy == 0 && dz == 0) continue;

                    // 获取邻居区块
                    Vector3Int neighborChunkPos = chunkPosition + new Vector3Int(dx, dy, dz);
                    if (!generator.TryGetChunk(neighborChunkPos, out Chunk neighborChunk) || !neighborChunk.Initialized)
                    {
                        continue; // 如果邻居不存在或未初始化，则跳过
                    }

                    Voxel[] neighborVoxels = neighborChunk.Voxels;

                    // --- 确定要复制的源区域和目标区域 ---
                    // 所有坐标都在区块本地的 padded 空间中

                    // 邻居区块中源数据区域的起始坐标
                    int srcX_start = (dx == 1) ? 1 : (dx == -1 ? logicalChunkSize.x : 1);
                    int srcY_start = (dy == 1) ? 1 : (dy == -1 ? logicalChunkSize.y : 1);
                    int srcZ_start = (dz == 1) ? 1 : (dz == -1 ? logicalChunkSize.z : 1);

                    // 邻居区块中源数据区域的结束坐标
                    int srcX_end = (dx == 1) ? 1 : (dx == -1 ? logicalChunkSize.x : logicalChunkSize.x);
                    int srcY_end = (dy == 1) ? 1 : (dy == -1 ? logicalChunkSize.y : logicalChunkSize.y);
                    int srcZ_end = (dz == 1) ? 1 : (dz == -1 ? logicalChunkSize.z : logicalChunkSize.z);

                    // 本区块中目标Padding区域的起始坐标
                    int dstX_start = (dx == 1) ? paddedChunkSize.x - 1 : (dx == -1 ? 0 : 1);
                    int dstY_start = (dy == 1) ? paddedChunkSize.y - 1 : (dy == -1 ? 0 : 1);
                    int dstZ_start = (dz == 1) ? paddedChunkSize.z - 1 : (dz == -1 ? 0 : 1);

                    // 循环遍历源数据区域，并将其复制到目标Padding区域
                    // 这是一个高效的内存块操作，避免了昂贵的坐标转换
                    for (int x = srcX_start; x <= srcX_end; x++)
                    {
                        for (int y = srcY_start; y <= srcY_end; y++)
                        {
                            for (int z = srcZ_start; z <= srcZ_end; z++)
                            {
                                // 计算目标坐标
                                int dstX = dstX_start + (x - srcX_start);
                                int dstY = dstY_start + (y - srcY_start);
                                int dstZ = dstZ_start + (z - srcZ_start);

                                int srcIndex = VoxelUtil.To1DIndex(new Vector3Int(x, y, z), paddedChunkSize);
                                int dstIndex = VoxelUtil.To1DIndex(new Vector3Int(dstX, dstY, dstZ), paddedChunkSize);

                                voxels[dstIndex] = neighborVoxels[srcIndex];
                            }
                        }
                    }
                }
            }
        }
    }

    IEnumerator UpdateMesh()
    {
        if (Updating)
            yield break;

        if (!generator.CanUpdate)
            yield break;

        generator.UpdatingChunks++;

        // 在构建网格之前，从邻居那里更新填充体素
        UpdatePaddingVoxels();

        meshData?.Dispose();
        meshData = new VoxelMeshBuilder.NativeMeshData(VoxelUtil.ToInt3(paddedChunkSize));

        // 1. 分派作业（非阻塞）
        // Pass Logic Chunk Size (32) to the jobs, as they now iterate 0..32 and handle padding lookups internally.
        // The voxels array itself is still Padded Size (34), which matches meshData.nativeVoxels.
        meshData.ScheduleMeshingJob(voxels, VoxelUtil.ToInt3(logicalChunkSize));

        // 2. 非阻塞地等待作业完成
        yield return new WaitUntil(() => meshData.jobHandle.IsCompleted);

        // 3. 作业完成后，安全地完成句柄并应用数据
        meshData.jobHandle.Complete();

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

            // Force synchronous update to prevent race conditions where mesh is cleared while baking
            SetSharedMesh(mesh);
            
            // if (argent)
            //     SetSharedMesh(mesh);
            // else
            //     VoxelColliderBuilder.Instance.Enqueue(this, mesh);
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