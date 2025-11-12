using System.Collections.Generic;
using OptIn.Voxel;
using Priority_Queue;
using UnityEngine;

public class TerrainGenerator : Singleton<TerrainGenerator>
{
    [SerializeField] Transform target;
    [SerializeField] Vector3Int chunkSize = Vector3Int.one * 32;
    [SerializeField] Vector2Int chunkSpawnSize = Vector2Int.one * 8;
    [SerializeField] Material chunkMaterial;
    [SerializeField] int maxGenerateChunksInFrame = 5;

    class ChunkNode : FastPriorityQueueNode
    {
        public Vector3Int chunkPosition;
    }

    Dictionary<Vector3Int, Chunk> chunks = new Dictionary<Vector3Int, Chunk>();
    Vector3Int lastTargetChunkPosition = new Vector3Int(int.MinValue, int.MaxValue, int.MinValue);
    FastPriorityQueue<ChunkNode> generateChunkQueue = new FastPriorityQueue<ChunkNode>(100000);
    int updatingChunks;
    public ComputeShader voxelComputeShader;

    public Vector3Int ChunkSize => chunkSize;
    public Material ChunkMaterial => chunkMaterial;

    public int UpdatingChunks
    {
        get => updatingChunks;
        set => updatingChunks = value;
    }

    public bool CanUpdate => updatingChunks <= maxGenerateChunksInFrame;

    void Awake()
    {
        VoxelMeshBuilder.InitializeShaderParameter();
    }

    void Update()
    {
        GenerateChunkByTargetPosition();
    }

    void LateUpdate()
    {
        ProcessGenerateChunkQueue();
    }

    void GenerateChunkByTargetPosition()
    {
        if (target == null) return;

        Vector3Int targetPosition = VoxelUtil.WorldToChunk(target.position, chunkSize);
        if (lastTargetChunkPosition == targetPosition) return;

        var toRemove = new List<ChunkNode>();
        foreach (ChunkNode chunkNode in generateChunkQueue)
        {
            Vector3Int deltaPosition = targetPosition - chunkNode.chunkPosition;
            if (Mathf.Abs(deltaPosition.x) > chunkSpawnSize.x ||
                Mathf.Abs(deltaPosition.y) > chunkSpawnSize.y ||
                Mathf.Abs(deltaPosition.z) > chunkSpawnSize.x)
            {
                toRemove.Add(chunkNode);
            }
            else
            {
                generateChunkQueue.UpdatePriority(chunkNode, (targetPosition - chunkNode.chunkPosition).sqrMagnitude);
            }
        }
        foreach (var node in toRemove)
        {
            if (generateChunkQueue.Contains(node))
                generateChunkQueue.Remove(node);
        }

        for (int x = targetPosition.x - chunkSpawnSize.x; x <= targetPosition.x + chunkSpawnSize.x; x++)
            for (int y = targetPosition.y - chunkSpawnSize.y; y <= targetPosition.y + chunkSpawnSize.y; y++)
                for (int z = targetPosition.z - chunkSpawnSize.x; z <= targetPosition.z + chunkSpawnSize.x; z++)
                {
                    Vector3Int chunkPosition = new Vector3Int(x, y, z);
                    if (chunks.ContainsKey(chunkPosition)) continue;

                    ChunkNode newNode = new ChunkNode { chunkPosition = chunkPosition };
                    if (generateChunkQueue.Contains(newNode)) continue;

                    generateChunkQueue.Enqueue(newNode, (targetPosition - chunkPosition).sqrMagnitude);
                }

        lastTargetChunkPosition = targetPosition;
    }

    void ProcessGenerateChunkQueue()
    {
        int numChunks = 0;
        while (generateChunkQueue.Count > 0 && numChunks < maxGenerateChunksInFrame)
        {
            Vector3Int chunkPosition = generateChunkQueue.Dequeue().chunkPosition;
            GenerateChunk(chunkPosition);
            numChunks++;
        }
    }

    Chunk GenerateChunk(Vector3Int chunkPosition)
    {
        if (chunks.ContainsKey(chunkPosition)) return chunks[chunkPosition];

        GameObject chunkGameObject = new GameObject(chunkPosition.ToString());
        chunkGameObject.transform.SetParent(transform);
        chunkGameObject.transform.position = VoxelUtil.ChunkToWorld(chunkPosition, chunkSize);

        Chunk newChunk = chunkGameObject.AddComponent<Chunk>();
        newChunk.Init(chunkPosition, this);
        newChunk.CanUpdate += () =>
        {
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    for (int z = -1; z <= 1; z++)
                    {
                        if (x == 0 && y == 0 && z == 0) continue;
                        if (chunks.TryGetValue(chunkPosition + new Vector3Int(x, y, z), out Chunk neighborChunk))
                        {
                            if (!neighborChunk.Initialized) return false;
                        }
                    }
            return true;
        };

        chunks.Add(chunkPosition, newChunk);
        return newChunk;
    }

    public bool GetChunk(Vector3 worldPosition, out Chunk chunk)
    {
        Vector3Int chunkPosition = VoxelUtil.WorldToChunk(worldPosition, chunkSize);
        return chunks.TryGetValue(chunkPosition, out chunk);
    }

    public bool GetVoxel(Vector3 worldPosition, out Voxel voxel)
    {
        if (GetChunk(worldPosition, out Chunk chunk))
        {
            Vector3Int gridPosition = VoxelUtil.WorldToGrid(worldPosition, chunk.transform.position, chunkSize);
            return chunk.GetVoxel(gridPosition, out voxel);
        }
        voxel = Voxel.Empty;
        return false;
    }

    public bool IsAir(Vector3 worldPosition)
    {
        if (GetVoxel(worldPosition, out Voxel voxel))
        {
            return voxel.IsAir;
        }
        return true;
    }

    public bool SetVoxel(Vector3 worldPosition, short voxelID)
    {
        Voxel voxel = new Voxel { voxelID = voxelID };
        if (voxelID <= 0) // Isosurface
        {
            voxel.Density = voxelID == 0 ? -1f : 1f;
        }
        return SetVoxel(worldPosition, voxel);
    }

    public bool SetVoxel(Vector3 worldPosition, Voxel voxel)
    {
        Vector3Int chunkPos = VoxelUtil.WorldToChunk(worldPosition, chunkSize);
        if (chunks.TryGetValue(chunkPos, out Chunk chunk))
        {
            Vector3Int gridPos = VoxelUtil.WorldToGrid(worldPosition, VoxelUtil.ChunkToWorld(chunkPos, chunkSize), chunkSize);
            if (chunk.SetVoxel(gridPos, voxel))
            {
                if (gridPos.x == 0 || gridPos.x == chunkSize.x - 1 ||
                    gridPos.y == 0 || gridPos.y == chunkSize.y - 1 ||
                    gridPos.z == 0 || gridPos.z == chunkSize.z - 1)
                {
                    NotifyNeighborChunks(worldPosition);
                }
                return true;
            }
        }
        return false;
    }

    public void ModifySphere(Vector3 center, float radius, float intensity, short materialId)
    {
        int radiusInt = Mathf.CeilToInt(radius);
        HashSet<Vector3Int> dirtyChunks = new HashSet<Vector3Int>();

        for (int x = -radiusInt; x <= radiusInt; x++)
            for (int y = -radiusInt; y <= radiusInt; y++)
                for (int z = -radiusInt; z <= radiusInt; z++)
                {
                    Vector3 delta = new Vector3(x, y, z);
                    float dist = delta.magnitude;
                    if (dist > radius) continue;

                    Vector3 worldPos = center + delta;
                    Vector3Int chunkPos = VoxelUtil.WorldToChunk(worldPos, chunkSize);

                    if (chunks.TryGetValue(chunkPos, out Chunk chunk))
                    {
                        Vector3Int gridPos = VoxelUtil.WorldToGrid(worldPos, VoxelUtil.ChunkToWorld(chunkPos, chunkSize), chunkSize);
                        if (chunk.GetVoxel(gridPos, out Voxel oldVoxel))
                        {
                            float densityChange = (1.0f - dist / radius) * intensity;
                            float newDensity = oldVoxel.Density + densityChange;

                            Voxel newVoxel = new Voxel();
                            newVoxel.Density = newDensity;

                            if (newDensity > 0)
                            {
                                // If old voxel was a block, keep its ID. Otherwise, use new material ID.
                                newVoxel.voxelID = oldVoxel.IsBlock ? oldVoxel.voxelID : materialId;
                            }
                            else
                            {
                                newVoxel.voxelID = 0; // Air
                            }

                            chunk.SetVoxel(gridPos, newVoxel);
                            dirtyChunks.Add(chunkPos);
                        }
                    }
                }

        foreach (var chunkPos in dirtyChunks)
        {
            if (chunks.TryGetValue(chunkPos, out var chunk))
            {
                chunk.NeighborChunkIsChanged();
                NotifyNeighborChunks(VoxelUtil.ChunkToWorld(chunkPos, chunkSize) + (Vector3)chunkSize * 0.5f);
            }
        }
    }

    private void NotifyNeighborChunks(Vector3 worldPosition)
    {
        Vector3Int centerChunkPos = VoxelUtil.WorldToChunk(worldPosition, chunkSize);
        for (int x = -1; x <= 1; x++)
            for (int y = -1; y <= 1; y++)
                for (int z = -1; z <= 1; z++)
                {
                    Vector3Int neighborPos = centerChunkPos + new Vector3Int(x, y, z);
                    if (chunks.TryGetValue(neighborPos, out Chunk neighborChunk))
                    {
                        neighborChunk.NeighborChunkIsChanged();
                    }
                }
    }

    public List<Voxel[]> GetNeighborVoxels(Vector3Int chunkPosition, int numNeighbor)
    {
        List<Voxel[]> neighborVoxels = new List<Voxel[]>();
        for (int x = chunkPosition.x - numNeighbor; x <= chunkPosition.x + numNeighbor; x++)
            for (int y = chunkPosition.y - numNeighbor; y <= chunkPosition.y + numNeighbor; y++)
                for (int z = chunkPosition.z - numNeighbor; z <= chunkPosition.z + numNeighbor; z++)
                {
                    if (chunks.TryGetValue(new Vector3Int(x, y, z), out Chunk chunk))
                    {
                        neighborVoxels.Add(chunk.Voxels);
                    }
                    else
                    {
                        neighborVoxels.Add(null);
                    }
                }
        return neighborVoxels;
    }
}