using System.Collections;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace OptIn.Voxel
{
    public static class VoxelMeshBuilder
    {
        public static void InitializeShaderParameter()
        {
            Shader.SetGlobalInt("_AtlasX", AtlasSize.x);
            Shader.SetGlobalInt("_AtlasY", AtlasSize.y);
            Shader.SetGlobalVector("_AtlasRec", new Vector4(1.0f / AtlasSize.x, 1.0f / AtlasSize.y));
        }

        public static readonly int2 AtlasSize = new int2(8, 8);

        public class NativeMeshData
        {
            NativeArray<Voxel> nativeVoxels;
            public NativeArray<GPUVertex> nativeVertices;
            public NativeArray<int> nativeIndices;
            public JobHandle jobHandle;
            NativeCounter counter;

            public NativeMeshData(int3 chunkSize)
            {
                int numVoxels = chunkSize.x * chunkSize.y * chunkSize.z;
                int maxVertices = 12 * numVoxels; 
                int maxIndices = 18 * numVoxels;

                nativeVoxels = new NativeArray<Voxel>(numVoxels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                nativeVertices = new NativeArray<GPUVertex>(maxVertices, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                nativeIndices = new NativeArray<int>(maxIndices, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                counter = new NativeCounter(Allocator.Persistent);
            }

            ~NativeMeshData()
            {
                jobHandle.Complete();
                Dispose();
            }

            public void Dispose()
            {
                jobHandle.Complete();
                if (nativeVoxels.IsCreated) nativeVoxels.Dispose();
                if (nativeVertices.IsCreated) nativeVertices.Dispose();
                if (nativeIndices.IsCreated) nativeIndices.Dispose();
                if (counter.IsCreated) counter.Dispose();
            }

            public void ScheduleMeshingJob(Voxel[] voxels, int3 chunkSize)
            {
                jobHandle.Complete();
                nativeVoxels.CopyFrom(voxels);
                counter.Count = 0;
                NativeCounter.Concurrent concurrentCounter = counter.ToConcurrent();

                // 1. Run Greedy Meshing for Blocks (ID > 0)
                VoxelGreedyMeshingJob greedyJob = new VoxelGreedyMeshingJob
                {
                    voxels = nativeVoxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = concurrentCounter,
                };

                // Update jobHandle immediately so if the next step fails, we still track this job
                jobHandle = greedyJob.Schedule();

                // 2. Run Dual Contouring for Smooth Surfaces (ID < 0)
                VoxelDualContouringJob dcJob = new VoxelDualContouringJob
                {
                    voxels = nativeVoxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = concurrentCounter,
                };

                // Chain dependency
                jobHandle = dcJob.Schedule(jobHandle);
                
                JobHandle.ScheduleBatchedJobs();
            }

            public void GetMeshInformation(out int verticeSize, out int indicesSize)
            {
                verticeSize = counter.Count * 4;
                indicesSize = counter.Count * 6;
            }
        }

        [BurstCompile]
        struct VoxelGreedyMeshingJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;

            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<GPUVertex> vertices;

            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<int> indices;

            public NativeCounter.Concurrent counter;

            struct Empty { }

            public void Execute()
            {
                for (int direction = 0; direction < 6; direction++)
                {
                    NativeHashMap<int3, Empty> visited = new NativeHashMap<int3, Empty>(
                        chunkSize[VoxelUtil.DirectionAlignedX[direction]] * chunkSize[VoxelUtil.DirectionAlignedY[direction]], 
                        Allocator.Temp);

                    for (int depth = 1; depth < chunkSize[VoxelUtil.DirectionAlignedZ[direction]] - 1; depth++)
                    {
                        for (int x = 1; x < chunkSize[VoxelUtil.DirectionAlignedX[direction]] - 1; x++)
                        {
                            for (int y = 1; y < chunkSize[VoxelUtil.DirectionAlignedY[direction]] - 1;)
                            {
                                int3 gridPosition = new int3 
                                { 
                                    [VoxelUtil.DirectionAlignedX[direction]] = x, 
                                    [VoxelUtil.DirectionAlignedY[direction]] = y, 
                                    [VoxelUtil.DirectionAlignedZ[direction]] = depth 
                                };

                                int index = VoxelUtil.To1DIndex(gridPosition, chunkSize);
                                Voxel voxel = voxels[index];

                                // STRICT SEPARATION: Only process Blocks (> 0)
                                if (voxel.voxelID <= 0)
                                {
                                    y++;
                                    continue;
                                }

                                if (visited.ContainsKey(gridPosition))
                                {
                                    y++;
                                    continue;
                                }

                                // Check neighbor for occlusion. For blocks, we check if neighbor is visible.
                                // If neighbor is NOT transparent (meaning it's solid), we cull.
                                int3 neighborPosition = gridPosition + VoxelUtil.VoxelDirectionOffsets[direction];
                                if (!TransparencyCheck(voxels, neighborPosition, chunkSize))
                                {
                                    y++;
                                    continue;
                                }

                                visited.TryAdd(gridPosition, new Empty());
                                int height;
                                for (height = 1; height + y < chunkSize[VoxelUtil.DirectionAlignedY[direction]] - 1; height++)
                                {
                                    int3 nextPosition = gridPosition;
                                    nextPosition[VoxelUtil.DirectionAlignedY[direction]] += height;

                                    if (visited.ContainsKey(nextPosition)) break;

                                    Voxel nextVoxel = voxels[VoxelUtil.To1DIndex(nextPosition, chunkSize)];
                                    if (nextVoxel.voxelID != voxel.voxelID) break; 

                                    int3 nextNeighborPos = nextPosition + VoxelUtil.VoxelDirectionOffsets[direction];
                                    if (!TransparencyCheck(voxels, nextNeighborPos, chunkSize)) break;

                                    visited.TryAdd(nextPosition, new Empty());
                                }

                                bool isDone = false;
                                int width;
                                for (width = 1; width + x < chunkSize[VoxelUtil.DirectionAlignedX[direction]] - 1; width++)
                                {
                                    for (int dy = 0; dy < height; dy++)
                                    {
                                        int3 nextPosition = gridPosition;
                                        nextPosition[VoxelUtil.DirectionAlignedX[direction]] += width;
                                        nextPosition[VoxelUtil.DirectionAlignedY[direction]] += dy;

                                        if (visited.ContainsKey(nextPosition))
                                        {
                                            isDone = true;
                                            break;
                                        }

                                        Voxel nextVoxel = voxels[VoxelUtil.To1DIndex(nextPosition, chunkSize)];
                                        if (nextVoxel.voxelID != voxel.voxelID) 
                                        {
                                            isDone = true;
                                            break;
                                        }

                                        int3 nextNeighborPos = nextPosition + VoxelUtil.VoxelDirectionOffsets[direction];
                                        if (!TransparencyCheck(voxels, nextNeighborPos, chunkSize))
                                        {
                                            isDone = true;
                                            break;
                                        }
                                    }

                                    if (isDone) break;

                                    for (int dy = 0; dy < height; dy++)
                                    {
                                        int3 nextPosition = gridPosition;
                                        nextPosition[VoxelUtil.DirectionAlignedX[direction]] += width;
                                        nextPosition[VoxelUtil.DirectionAlignedY[direction]] += dy;
                                        visited.TryAdd(nextPosition, new Empty());
                                    }
                                }

                                AddQuadByDirection(direction, voxel.GetMaterialID(), width, height, gridPosition, counter.Increment(), vertices, indices);
                                y += height;
                            }
                        }
                        visited.Clear();
                    }
                    visited.Dispose();
                }
            }
        }

        [BurstCompile]
        struct VoxelDualContouringJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<GPUVertex> vertices;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<int> indices;
            public NativeCounter.Concurrent counter;

            private Voxel GetVoxelOrEmpty(int3 pos)
            {
                return VoxelUtil.BoundaryCheck(pos, chunkSize) ? voxels[VoxelUtil.To1DIndex(pos, chunkSize)] : Voxel.Empty;
            }

            private bool SignChanged(Voxel v1, Voxel v2) 
            {
                // Only consider isosurface voxels (and air) for this check to avoid blending with blocks
                if (v1.IsBlock || v2.IsBlock) return false;
                
                // Classic DC sign change: One is inside (density > 0), one is outside (density <= 0)
                // Note: User defines Negative IDs as Smooth. 
                // However, Density property handles the internal float conversion.
                // We assume Air (ID 0) has Density <= 0. Smooth (ID < 0) has Density based on metadata.
                return (v1.Density > 0) != (v2.Density > 0);
            }

            private float3 CalculateFeaturePoint(int3 pos)
            {
                float3 pointSum = float3.zero;
                int crossings = 0;
                for (int i = 0; i < 12; i++)
                {
                    Voxel v1 = GetVoxelOrEmpty(pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 0]]);
                    Voxel v2 = GetVoxelOrEmpty(pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 1]]);
                    
                    // STRICT CHECK: Only process if both are NOT blocks.
                    if (!v1.IsBlock && !v2.IsBlock && SignChanged(v1, v2))
                    {
                        float t = math.unlerp(v1.Density, v2.Density, 0f);
                        if (!float.IsFinite(t)) t = 0.5f;
                        pointSum += math.lerp(pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 0]], pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 1]], t);
                        crossings++;
                    }
                }
                return crossings > 0 ? pointSum / crossings : (float3)pos + 0.5f;
            }

            private float GetDensityForGradient(int3 pos, int3 chunkSize, [ReadOnly] NativeArray<Voxel> voxelData)
            {
                // Fallback for gradient calculation: treat blocks as full density or empty?
                // For smooth meshing, we only care about smooth voxels.
                // If we hit a block, return 0 (outside) or 1 (inside)? 
                // To isolate, maybe just return density. Blocks have density 1.
                // But we shouldn't really be calculating gradients ON blocks.
                return voxelData[VoxelUtil.To1DIndex(pos, chunkSize)].Density;
            }

            private float3 CalculatePaddedGradient(int3 pos, int3 chunkSize, [ReadOnly] NativeArray<Voxel> voxelData)
            {
                 float dx, dy, dz;
                // Standard central difference
                if (pos.x == 0) dx = GetDensityForGradient(pos + new int3(1, 0, 0), chunkSize, voxelData) - GetDensityForGradient(pos, chunkSize, voxelData);
                else if (pos.x == chunkSize.x - 1) dx = GetDensityForGradient(pos, chunkSize, voxelData) - GetDensityForGradient(pos - new int3(1, 0, 0), chunkSize, voxelData);
                else dx = (GetDensityForGradient(pos + new int3(1, 0, 0), chunkSize, voxelData) - GetDensityForGradient(pos - new int3(1, 0, 0), chunkSize, voxelData)) * 0.5f;

                if (pos.y == 0) dy = GetDensityForGradient(pos + new int3(0, 1, 0), chunkSize, voxelData) - GetDensityForGradient(pos, chunkSize, voxelData);
                else if (pos.y == chunkSize.y - 1) dy = GetDensityForGradient(pos, chunkSize, voxelData) - GetDensityForGradient(pos - new int3(0, 1, 0), chunkSize, voxelData);
                else dy = (GetDensityForGradient(pos + new int3(0, 1, 0), chunkSize, voxelData) - GetDensityForGradient(pos - new int3(0, 1, 0), chunkSize, voxelData)) * 0.5f;

                if (pos.z == 0) dz = GetDensityForGradient(pos + new int3(0, 0, 1), chunkSize, voxelData) - GetDensityForGradient(pos, chunkSize, voxelData);
                else if (pos.z == chunkSize.z - 1) dz = GetDensityForGradient(pos, chunkSize, voxelData) - GetDensityForGradient(pos - new int3(0, 0, 1), chunkSize, voxelData);
                else dz = (GetDensityForGradient(pos + new int3(0, 0, 1), chunkSize, voxelData) - GetDensityForGradient(pos - new int3(0, 0, 1), chunkSize, voxelData)) * 0.5f;

                float3 grad = new float3(dx, dy, dz);
                return math.normalizesafe(-grad, new float3(0, 1, 0));
            }

            public void Execute()
            {
                int numVoxels = chunkSize.x * chunkSize.y * chunkSize.z;
                var gradients = new NativeArray<float3>(numVoxels, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

                // Pass 1: gradients
                for (int x = 0; x < chunkSize.x; x++)
                    for (int y = 0; y < chunkSize.y; y++)
                        for (int z = 0; z < chunkSize.z; z++)
                        {
                            var pos = new int3(x, y, z);
                            int index = VoxelUtil.To1DIndex(pos, chunkSize);
                            // Optim: Only calculate gradient if this or neighbor is smooth surface? 
                            // For simplicity, calc for all.
                            gradients[index] = CalculatePaddedGradient(pos, chunkSize, voxels);
                        }

                // Pass 2: Mesh
                for (int x = 1; x < chunkSize.x - 1; x++)
                    for (int y = 1; y < chunkSize.y - 1; y++)
                        for (int z = 1; z < chunkSize.z - 1; z++)
                        {
                            var pos = new int3(x, y, z);
                            var voxel = GetVoxelOrEmpty(pos);

                            if (voxel.IsBlock) continue; // Skip blocks

                            for (int axis = 0; axis < 3; axis++)
                            {
                                var neighbor = GetVoxelOrEmpty(pos + VoxelUtil.DC_AXES[axis]);
                                
                                if (neighbor.IsBlock) continue; // Skip if neighbor is block

                                // STRICT SEPARATION: Only smooth mesh if Isosurface (<0 or 0) involved
                                if (SignChanged(voxel, neighbor))
                                {
                                    int quadIndex = counter.Increment();
                                    // Use absolute ID for material lookup
                                    ushort materialId = voxel.Density > 0 ? voxel.GetMaterialID() : neighbor.GetMaterialID();

                                    for (int i = 0; i < 4; i++)
                                    {
                                        var cornerPos = pos + VoxelUtil.DC_ADJACENT[axis, i];
                                        vertices[quadIndex * 4 + i] = new GPUVertex
                                        {
                                            position = CalculateFeaturePoint(cornerPos) - 1,
                                            normal = gradients[VoxelUtil.To1DIndex(cornerPos, chunkSize)],
                                            uv = new float4(0, 0, materialId, 0)
                                        };
                                    }

                                    int vertIndex = quadIndex * 4;
                                    if (voxel.Density > 0)
                                    {
                                        indices[quadIndex * 6 + 0] = vertIndex + 0;
                                        indices[quadIndex * 6 + 1] = vertIndex + 1;
                                        indices[quadIndex * 6 + 2] = vertIndex + 2;
                                        indices[quadIndex * 6 + 3] = vertIndex + 0;
                                        indices[quadIndex * 6 + 4] = vertIndex + 2;
                                        indices[quadIndex * 6 + 5] = vertIndex + 3;
                                    }
                                    else
                                    {
                                        indices[quadIndex * 6 + 0] = vertIndex + 0;
                                        indices[quadIndex * 6 + 1] = vertIndex + 2;
                                        indices[quadIndex * 6 + 2] = vertIndex + 1;
                                        indices[quadIndex * 6 + 3] = vertIndex + 0;
                                        indices[quadIndex * 6 + 4] = vertIndex + 3;
                                        indices[quadIndex * 6 + 5] = vertIndex + 2;
                                    }
                                }
                            }
                        }
                gradients.Dispose();
            }
        }

        public static bool TransparencyCheck(NativeArray<Voxel> voxels, int3 position, int3 chunkSize)
        {
            if (!VoxelUtil.BoundaryCheck(position, chunkSize))
                return false;

            // For culling, we consider a neighbor "transparent" if it is NOT a block (Solid).
            // This means smooth voxels (negative ID) are treated as transparent neighbors to blocks,
            // ensuring the block faces adjacent to smooth terrain are generated (closed mesh).
            var v = voxels[VoxelUtil.To1DIndex(position, chunkSize)];
            return !v.IsBlock; 
        }

        static void AddQuadByDirection(int direction, ushort materialID, float width, float height, int3 gridPosition, int quadIndex, NativeArray<GPUVertex> vertices, NativeArray<int> indices)
        {
            int vertexStart = quadIndex * 4;
            
            int atlasIndex = materialID * 6 + direction;
            int2 atlasPosition = new int2(atlasIndex % AtlasSize.x, atlasIndex / AtlasSize.x);

            for (int i = 0; i < 4; i++)
            {
                float3 pos = VoxelUtil.CubeVertices[VoxelUtil.CubeFaces[i + direction * 4]];
                pos[VoxelUtil.DirectionAlignedX[direction]] *= width;
                pos[VoxelUtil.DirectionAlignedY[direction]] *= height;

                float4 uv = new float4(
                    VoxelUtil.CubeUVs[i].x * width, 
                    VoxelUtil.CubeUVs[i].y * height, 
                    atlasPosition.x, 
                    atlasPosition.y
                );

                vertices[vertexStart + i] = new GPUVertex
                {
                    position = pos + gridPosition - 1, // Subtract padding (1) from position
                    normal = VoxelUtil.VoxelDirectionOffsets[direction],
                    uv = uv
                };
            }

            int indexStart = quadIndex * 6;
            for (int i = 0; i < 6; i++)
            {
                indices[indexStart + i] = VoxelUtil.CubeIndices[i + direction * 6] + vertexStart;
            }
        }
    }
}
