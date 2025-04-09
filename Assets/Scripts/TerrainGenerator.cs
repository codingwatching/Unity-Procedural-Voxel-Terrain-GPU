﻿using System.Collections;
using System.Collections.Generic;
using OptIn.Voxel;
using Priority_Queue;
using UnityEngine;

public class TerrainGenerator : Singleton<TerrainGenerator>
{
    [SerializeField] Transform target;
    [SerializeField] Vector3Int chunkSize = Vector3Int.one * 32;
    [SerializeField] Vector2Int chunkSpawnSize = Vector2Int.one;
    [SerializeField] Material chunkMaterial;
    [SerializeField] int maxGenerateChunksInFrame = 5;
    [SerializeField] VoxelMeshBuilder.SimplifyingMethod simplifyingMethod;

    class ChunkNode : FastPriorityQueueNode
    {
        public Vector3Int chunkPosition;
    }

    Dictionary<Vector3Int, Chunk> chunks = new Dictionary<Vector3Int, Chunk>();
    Vector3Int lastTargetChunkPosition = new Vector3Int(int.MinValue, int.MaxValue, int.MinValue);
    FastPriorityQueue<ChunkNode> generateChunkQueue = new FastPriorityQueue<ChunkNode>(100000);
    int updatingChunks;
    public ComputeShader voxelComputeShader;
    public ComputeShader meshComputeShader;

    public Vector3Int ChunkSize => chunkSize;
    public Material ChunkMaterial => chunkMaterial;
    public VoxelMeshBuilder.SimplifyingMethod SimplifyingMethod => simplifyingMethod;

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
        if (target == null)
            return;

        Vector3Int targetPosition = VoxelUtil.WorldToChunk(target.position, chunkSize);

        if (lastTargetChunkPosition == targetPosition)
            return;

        foreach (ChunkNode chunkNode in generateChunkQueue)
        {
            Vector3Int deltaPosition = targetPosition - chunkNode.chunkPosition;
            if (chunkSpawnSize.x < Mathf.Abs(deltaPosition.x) ||
                chunkSpawnSize.y < Mathf.Abs(deltaPosition.y) ||
                chunkSpawnSize.x < Mathf.Abs(deltaPosition.z))
            {
                generateChunkQueue.Remove(chunkNode);
                continue;
            }

            generateChunkQueue.UpdatePriority(chunkNode, (targetPosition - chunkNode.chunkPosition).sqrMagnitude);
        }

        for (int x = targetPosition.x - chunkSpawnSize.x; x <= targetPosition.x + chunkSpawnSize.x; x++)
        {
            for (int y = targetPosition.y - chunkSpawnSize.y; y <= targetPosition.y + chunkSpawnSize.y; y++)
            {
                for (int z = targetPosition.z - chunkSpawnSize.x; z <= targetPosition.z + chunkSpawnSize.x; z++)
                {
                    Vector3Int chunkPosition = new Vector3Int(x, y, z);
                    if (chunks.ContainsKey(chunkPosition))
                        continue;

                    ChunkNode newNode = new ChunkNode { chunkPosition = chunkPosition };

                    if (generateChunkQueue.Contains(newNode))
                        continue;

                    generateChunkQueue.Enqueue(newNode, (targetPosition - chunkPosition).sqrMagnitude);
                }
            }
        }

        lastTargetChunkPosition = targetPosition;
    }

    void ProcessGenerateChunkQueue()
    {
        int numChunks = 0;
        while (generateChunkQueue.Count != 0)
        {
            if (numChunks >= maxGenerateChunksInFrame)
                return;

            Vector3Int chunkPosition = generateChunkQueue.Dequeue().chunkPosition;
            GenerateChunk(chunkPosition);
            numChunks++;
        }
    }

    Chunk GenerateChunk(Vector3Int chunkPosition)
    {
        if (chunks.ContainsKey(chunkPosition))
            return chunks[chunkPosition];

        GameObject chunkGameObject = new GameObject(chunkPosition.ToString());
        chunkGameObject.transform.SetParent(transform);
        chunkGameObject.transform.position = VoxelUtil.ChunkToWorld(chunkPosition, chunkSize);

        Chunk newChunk = chunkGameObject.AddComponent<Chunk>();
        newChunk.Init(chunkPosition, this);
        newChunk.CanUpdate += delegate
        {
            for (int x = chunkPosition.x - 1; x <= chunkPosition.x + 1; x++)
            {
                for (int y = chunkPosition.y - 1; y <= chunkPosition.y + 1; y++)
                {
                    for (int z = chunkPosition.z - 1; z <= chunkPosition.z + 1; z++)
                    {
                        Vector3Int neighborChunkPosition = new Vector3Int(x, y, z);
                        if (chunks.TryGetValue(neighborChunkPosition, out Chunk neighborChunk))
                        {
                            if (!neighborChunk.Initialized)
                            {
                                return false;
                            }
                        }
                        else
                        {
                            return false;
                        }
                    }
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
            Vector3Int chunkPosition = VoxelUtil.WorldToChunk(worldPosition, chunkSize);
            Vector3Int gridPosition = VoxelUtil.WorldToGrid(worldPosition, chunkPosition, chunkSize);
            if (chunk.GetVoxel(gridPosition, out voxel))
                return true;
        }

        voxel = Voxel.Empty;
        return false;
    }

    public bool IsAir(Vector3 worldPosition)
    {
        if (GetVoxel(worldPosition, out Voxel voxel))
        {
            // 根据等价体素定义，Air 为 texId == 0 或密度不足
            return voxel.IsAir();
        }

        return false;
    }

    public bool SetVoxel(Vector3 worldPosition, ushort type)
    {
        if (GetChunk(worldPosition, out Chunk chunk))
        {
            Vector3Int chunkPosition = VoxelUtil.WorldToChunk(worldPosition, chunkSize);
            Vector3Int gridPosition = VoxelUtil.WorldToGrid(worldPosition, chunkPosition, chunkSize);
            if (chunk.SetVoxel(gridPosition, type))
            {
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        for (int z = -1; z <= 1; z++)
                        {
                            if (VoxelUtil.BoundaryCheck(gridPosition + new Vector3Int(x, y, z), chunkSize))
                                continue;

                            Vector3Int neighborChunkPosition = VoxelUtil.WorldToChunk(worldPosition + new Vector3(x, y, z), chunkSize);
                            if (chunkPosition == neighborChunkPosition)
                                continue;

                            if (chunks.TryGetValue(neighborChunkPosition, out Chunk neighborChunk))
                            {
                                neighborChunk.NeighborChunkIsChanged();
                            }
                        }
                    }
                }

                return true;
            }
        }
        return false;
    }

    public List<Voxel[]> GetNeighborVoxels(Vector3Int chunkPosition, int numNeighbor)
    {
        List<Voxel[]> neighborVoxels = new List<Voxel[]>();

        for (int x = chunkPosition.x - numNeighbor; x <= chunkPosition.x + numNeighbor; x++)
        {
            for (int y = chunkPosition.y - numNeighbor; y <= chunkPosition.y + numNeighbor; y++)
            {
                for (int z = chunkPosition.z - numNeighbor; z <= chunkPosition.z + numNeighbor; z++)
                {
                    Vector3Int neighborChunkPosition = new Vector3Int(x, y, z);
                    if (chunks.TryGetValue(neighborChunkPosition, out Chunk chunk))
                    {
                        neighborVoxels.Add(chunk.Voxels);
                    }
                    else
                    {
                        neighborVoxels.Add(null);
                    }
                }
            }
        }

        return neighborVoxels;
    }
}
