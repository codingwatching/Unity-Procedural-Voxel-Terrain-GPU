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

        public enum SimplifyingMethod
        {
            Culling,
            GreedyOnlyHeight,
            Greedy,
            DualContouring,
        };

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
                if (nativeVoxels.IsCreated) nativeVoxels.Dispose();
                if (nativeVertices.IsCreated) nativeVertices.Dispose();
                if (nativeIndices.IsCreated) nativeIndices.Dispose();
                if (counter.IsCreated) counter.Dispose();
            }

            public IEnumerator ScheduleMeshingJob(Voxel[] voxels, int3 chunkSize, SimplifyingMethod method, bool argent = false)
            {
                nativeVoxels.CopyFrom(voxels);
                counter.Count = 0;

                // 强制使用已修复的、包含正确分离逻辑的DualContouringJob
                ScheduleDualContouringJob(nativeVoxels, chunkSize);

                yield return new WaitUntil(() => jobHandle.IsCompleted || argent);
                jobHandle.Complete();
            }

            public void GetMeshInformation(out int verticeSize, out int indicesSize)
            {
                verticeSize = counter.Count * 4;
                indicesSize = counter.Count * 6;
            }

            void ScheduleCullingJob(NativeArray<Voxel> voxels, int3 chunkSize)
            {
                VoxelCullingJob job = new VoxelCullingJob
                {
                    voxels = voxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = counter,
                };
                jobHandle = job.Schedule();
                JobHandle.ScheduleBatchedJobs();
            }

            void ScheduleDualContouringJob(NativeArray<Voxel> voxels, int3 chunkSize)
            {
                VoxelDualContouringJob job = new VoxelDualContouringJob
                {
                    voxels = voxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = counter.ToConcurrent(),
                };
                jobHandle = job.Schedule();
                JobHandle.ScheduleBatchedJobs();
            }

            void ScheduleGreedyOnlyHeightJob(NativeArray<Voxel> voxels, int3 chunkSize)
            {
                VoxelGreedyMeshingOnlyHeightJob job = new VoxelGreedyMeshingOnlyHeightJob
                {
                    voxels = voxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = counter,
                };
                jobHandle = job.Schedule();
                JobHandle.ScheduleBatchedJobs();
            }

            void ScheduleGreedyJob(NativeArray<Voxel> voxels, int3 chunkSize)
            {
                VoxelGreedyMeshingJob job = new VoxelGreedyMeshingJob
                {
                    voxels = voxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = counter,
                };
                jobHandle = job.Schedule();
                JobHandle.ScheduleBatchedJobs();
            }
        }

        static bool IsVoxelSolid(NativeArray<Voxel> voxels, int3 position, int3 chunkSize)
        {
            if (!VoxelUtil.BoundaryCheck(position, chunkSize))
                return false;
            return voxels[VoxelUtil.To1DIndex(position, chunkSize)].IsSolid;
        }

        [BurstCompile]
        struct VoxelCullingJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<GPUVertex> vertices;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<int> indices;
            [WriteOnly] public NativeCounter counter;

            public void Execute()
            {
                for (int x = 0; x < chunkSize.x; x++)
                    for (int y = 0; y < chunkSize.y; y++)
                        for (int z = 0; z < chunkSize.z; z++)
                        {
                            int3 gridPosition = new int3(x, y, z);
                            Voxel voxel = voxels[VoxelUtil.To1DIndex(gridPosition, chunkSize)];

                            if (!voxel.IsBlock) continue;

                            for (int direction = 0; direction < 6; direction++)
                            {
                                int3 neighborPosition = gridPosition + VoxelUtil.VoxelDirectionOffsets[direction];
                                if (IsVoxelSolid(voxels, neighborPosition, chunkSize)) continue;

                                AddQuadByDirection(direction, voxel.GetMaterialID(), 1.0f, 1.0f, gridPosition, counter.Increment(), vertices, indices);
                            }
                        }
            }
        }

        [BurstCompile]
        struct VoxelDualContouringJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<GPUVertex> vertices;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<int> indices;
            public NativeCounter.Concurrent counter;

            private Voxel GetVoxelOrEmpty(int3 pos)
            {
                return VoxelUtil.BoundaryCheck(pos, chunkSize) ? voxels[VoxelUtil.To1DIndex(pos, chunkSize)] : Voxel.Empty;
            }

            private bool SignChanged(Voxel v1, Voxel v2) => v1.Density > 0 != v2.Density > 0;

            private float3 CalculateFeaturePoint(int3 pos)
            {
                float3 pointSum = float3.zero;
                int crossings = 0;
                for (int i = 0; i < 12; i++)
                {
                    Voxel v1 = GetVoxelOrEmpty(pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 0]]);
                    Voxel v2 = GetVoxelOrEmpty(pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 1]]);
                    if (SignChanged(v1, v2))
                    {
                        float t = math.unlerp(v1.Density, v2.Density, 0f);
                        if (!float.IsFinite(t)) t = 0.5f;
                        pointSum += math.lerp(pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 0]], pos + VoxelUtil.DC_VERT[VoxelUtil.DC_EDGE[i, 1]], t);
                        crossings++;
                    }
                }
                return crossings > 0 ? pointSum / crossings : (float3)pos + 0.5f;
            }

            private float3 CalculateGradient(int3 pos)
            {
                float dx = GetVoxelOrEmpty(pos + new int3(1, 0, 0)).Density - GetVoxelOrEmpty(pos - new int3(1, 0, 0)).Density;
                float dy = GetVoxelOrEmpty(pos + new int3(0, 1, 0)).Density - GetVoxelOrEmpty(pos - new int3(0, 1, 0)).Density;
                float dz = GetVoxelOrEmpty(pos + new int3(0, 0, 1)).Density - GetVoxelOrEmpty(pos - new int3(0, 0, 1)).Density;
                float3 grad = new float3(dx, dy, dz);
                return math.normalizesafe(grad, -grad);
            }

            public void Execute()
            {
                for (int x = 0; x < chunkSize.x; x++)
                    for (int y = 0; y < chunkSize.y; y++)
                        for (int z = 0; z < chunkSize.z; z++)
                        {
                            var pos = new int3(x, y, z);
                            var voxel = GetVoxelOrEmpty(pos);

                            if (voxel.IsBlock)
                            {
                                // --- PATH 1: BLOCK VOXEL ---
                                // 使用原始的Culling算法，只生成暴露在外的面
                                for (int direction = 0; direction < 6; direction++)
                                {
                                    Voxel neighborVoxel = GetVoxelOrEmpty(pos + VoxelUtil.VoxelDirectionOffsets[direction]);
                                    // 如果邻居不是固体（空气或密度为负的平滑体素），则生成一个面
                                    if (!neighborVoxel.IsSolid)
                                    {
                                        AddQuadByDirection(direction, voxel.GetMaterialID(), 1.0f, 1.0f, pos, counter.Increment(), vertices, indices);
                                    }
                                }
                            }
                            else // IsIsosurface
                            {
                                // --- PATH 2: SMOOTH VOXEL (包括空气) ---
                                // 使用Dual Contouring算法，只在平滑体素之间生成面
                                for (int axis = 0; axis < 3; axis++)
                                {
                                    var neighbor = GetVoxelOrEmpty(pos + VoxelUtil.DC_AXES[axis]);

                                    // 关键修复：只在两个平滑体素之间进行符号检查，杜绝与方块的交互
                                    if (neighbor.IsIsosurface && SignChanged(voxel, neighbor))
                                    {
                                        int quadIndex = counter.Increment();
                                        ushort materialId = voxel.Density > 0 ? voxel.GetMaterialID() : neighbor.GetMaterialID();

                                        for (int i = 0; i < 4; i++)
                                        {
                                            var cornerPos = pos + VoxelUtil.DC_ADJACENT[axis, i];
                                            vertices[quadIndex * 4 + i] = new GPUVertex
                                            {
                                                position = CalculateFeaturePoint(cornerPos),
                                                normal = CalculateGradient(cornerPos),
                                                uv = new float4(0, 0, materialId, 0)
                                            };
                                        }

                                        int vertIndex = quadIndex * 4;
                                        indices[quadIndex * 6 + 0] = vertIndex + 0;
                                        indices[quadIndex * 6 + 1] = vertIndex + 1;
                                        indices[quadIndex * 6 + 2] = vertIndex + 2;
                                        indices[quadIndex * 6 + 3] = vertIndex + 0;
                                        indices[quadIndex * 6 + 4] = vertIndex + 2;
                                        indices[quadIndex * 6 + 5] = vertIndex + 3;
                                    }
                                }
                            }
                        }
            }
        }

        [BurstCompile]
        struct VoxelGreedyMeshingOnlyHeightJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<GPUVertex> vertices;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<int> indices;
            [WriteOnly] public NativeCounter counter;

            public void Execute()
            {
                for (int direction = 0; direction < 6; direction++)
                    for (int depth = 0; depth < chunkSize[VoxelUtil.DirectionAlignedZ[direction]]; depth++)
                        for (int x = 0; x < chunkSize[VoxelUtil.DirectionAlignedX[direction]]; x++)
                            for (int y = 0; y < chunkSize[VoxelUtil.DirectionAlignedY[direction]];)
                            {
                                int3 gridPosition = new int3 { [VoxelUtil.DirectionAlignedX[direction]] = x, [VoxelUtil.DirectionAlignedY[direction]] = y, [VoxelUtil.DirectionAlignedZ[direction]] = depth };
                                Voxel voxel = voxels[VoxelUtil.To1DIndex(gridPosition, chunkSize)];

                                if (!voxel.IsBlock) { y++; continue; }

                                int3 neighborPosition = gridPosition + VoxelUtil.VoxelDirectionOffsets[direction];
                                if (IsVoxelSolid(voxels, neighborPosition, chunkSize)) { y++; continue; }

                                int height;
                                for (height = 1; height + y < chunkSize[VoxelUtil.DirectionAlignedY[direction]]; height++)
                                {
                                    int3 nextPosition = gridPosition;
                                    nextPosition[VoxelUtil.DirectionAlignedY[direction]] += height;
                                    Voxel nextVoxel = voxels[VoxelUtil.To1DIndex(nextPosition, chunkSize)];
                                    if (nextVoxel.voxelID != voxel.voxelID) break;
                                }
                                AddQuadByDirection(direction, voxel.GetMaterialID(), 1.0f, height, gridPosition, counter.Increment(), vertices, indices);
                                y += height;
                            }
            }
        }

        [BurstCompile]
        struct VoxelGreedyMeshingJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<GPUVertex> vertices;
            [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<int> indices;
            [WriteOnly] public NativeCounter counter;
            struct Empty { }

            public void Execute()
            {
                for (int direction = 0; direction < 6; direction++)
                {
                    var hashMap = new NativeParallelHashMap<int3, Empty>(chunkSize[VoxelUtil.DirectionAlignedX[direction]] * chunkSize[VoxelUtil.DirectionAlignedY[direction]], Allocator.Temp);
                    for (int depth = 0; depth < chunkSize[VoxelUtil.DirectionAlignedZ[direction]]; depth++)
                    {
                        for (int x = 0; x < chunkSize[VoxelUtil.DirectionAlignedX[direction]]; x++)
                        {
                            for (int y = 0; y < chunkSize[VoxelUtil.DirectionAlignedY[direction]];)
                            {
                                int3 gridPosition = new int3 { [VoxelUtil.DirectionAlignedX[direction]] = x, [VoxelUtil.DirectionAlignedY[direction]] = y, [VoxelUtil.DirectionAlignedZ[direction]] = depth };
                                Voxel voxel = voxels[VoxelUtil.To1DIndex(gridPosition, chunkSize)];

                                if (!voxel.IsBlock || hashMap.ContainsKey(gridPosition)) { y++; continue; }

                                if (IsVoxelSolid(voxels, gridPosition + VoxelUtil.VoxelDirectionOffsets[direction], chunkSize)) { y++; continue; }

                                hashMap.TryAdd(gridPosition, new Empty());

                                int height;
                                for (height = 1; height + y < chunkSize[VoxelUtil.DirectionAlignedY[direction]]; height++)
                                {
                                    int3 nextPosition = gridPosition;
                                    nextPosition[VoxelUtil.DirectionAlignedY[direction]] += height;
                                    Voxel nextVoxel = voxels[VoxelUtil.To1DIndex(nextPosition, chunkSize)];
                                    if (nextVoxel.voxelID != voxel.voxelID || hashMap.ContainsKey(nextPosition)) break;
                                    hashMap.TryAdd(nextPosition, new Empty());
                                }

                                bool isDone = false;
                                int width;
                                for (width = 1; width + x < chunkSize[VoxelUtil.DirectionAlignedX[direction]]; width++)
                                {
                                    for (int dy = 0; dy < height; dy++)
                                    {
                                        int3 nextPosition = gridPosition;
                                        nextPosition[VoxelUtil.DirectionAlignedX[direction]] += width;
                                        nextPosition[VoxelUtil.DirectionAlignedY[direction]] += dy;
                                        Voxel nextVoxel = voxels[VoxelUtil.To1DIndex(nextPosition, chunkSize)];
                                        if (nextVoxel.voxelID != voxel.voxelID || hashMap.ContainsKey(nextPosition))
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
                                        hashMap.TryAdd(nextPosition, new Empty());
                                    }
                                }

                                AddQuadByDirection(direction, voxel.GetMaterialID(), width, height, gridPosition, counter.Increment(), vertices, indices);
                                y += height;
                            }
                        }
                        hashMap.Clear();
                    }
                    hashMap.Dispose();
                }
            }
        }

        static void AddQuadByDirection(int direction, ushort materialID, float width, float height, int3 gridPosition, int quadIndex, NativeArray<GPUVertex> vertices, NativeArray<int> indices)
        {
            int vertexStart = quadIndex * 4;
            for (int i = 0; i < 4; i++)
            {
                float3 pos = VoxelUtil.CubeVertices[VoxelUtil.CubeFaces[i + direction * 4]];
                pos[VoxelUtil.DirectionAlignedX[direction]] *= width;
                pos[VoxelUtil.DirectionAlignedY[direction]] *= height;

                int atlasIndex = materialID * 6 + direction;
                int2 atlasPosition = new int2(atlasIndex % AtlasSize.x, atlasIndex / AtlasSize.x);

                vertices[vertexStart + i] = new GPUVertex
                {
                    position = pos + gridPosition,
                    normal = VoxelUtil.VoxelDirectionOffsets[direction],
                    uv = new float4(VoxelUtil.CubeUVs[i].x * width, VoxelUtil.CubeUVs[i].y * height, atlasPosition.x, atlasPosition.y)
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