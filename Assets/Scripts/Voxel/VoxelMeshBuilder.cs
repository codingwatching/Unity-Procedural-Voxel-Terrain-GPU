using System.Collections;
using OptIn.Voxel.Utils;
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
            GPUCulling,
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
                if (nativeVoxels.IsCreated)
                    nativeVoxels.Dispose();

                if (nativeVertices.IsCreated)
                    nativeVertices.Dispose();

                if (nativeIndices.IsCreated)
                    nativeIndices.Dispose();

                if (counter.IsCreated)
                    counter.Dispose();
            }

            public IEnumerator ScheduleMeshingJob(Voxel[] voxels, int3 chunkSize, SimplifyingMethod method, bool argent = false)
            {
                nativeVoxels.CopyFrom(voxels);
                switch (method)
                {
                    case SimplifyingMethod.Culling:
                        ScheduleCullingJob(nativeVoxels, chunkSize);
                        break;
                }

                yield return new WaitUntil(() =>
                {
                    return jobHandle.IsCompleted || argent;
                });

                jobHandle.Complete();
            }

            public void GetMeshInformation(out int verticeSize, out int indicesSize)
            {
                verticeSize = counter.Count * 4;
                indicesSize = counter.Count * 6;
            }

            void ScheduleCullingJob(NativeArray<Voxel> voxels, int3 chunkSize)
            {
                VoxelCullingJob voxelCullingJob = new VoxelCullingJob
                {
                    voxels = voxels,
                    chunkSize = chunkSize,
                    vertices = nativeVertices,
                    indices = nativeIndices,
                    counter = counter,
                };

                jobHandle = voxelCullingJob.Schedule();
                JobHandle.ScheduleBatchedJobs();
            }
        }

        [BurstCompile]
        struct VoxelCullingJob : IJob
        {
            [ReadOnly] public NativeArray<Voxel> voxels;
            [ReadOnly] public int3 chunkSize;

            [NativeDisableParallelForRestriction]
            [WriteOnly]
            public NativeArray<GPUVertex> vertices;

            [NativeDisableParallelForRestriction]
            [WriteOnly]
            public NativeArray<int> indices;

            [WriteOnly] public NativeCounter counter;

            public void Execute()
            {
                for (int x = 0; x < chunkSize.x; x++)
                {
                    for (int y = 0; y < chunkSize.y; y++)
                    {
                        for (int z = 0; z < chunkSize.z; z++)
                        {
                            int3 gridPosition = new int3(x, y, z);
                            int index = VoxelUtil.To1DIndex(gridPosition, chunkSize);

                            Voxel voxel = voxels[index];

                            if (voxel.data == Voxel.VoxelType.Air)
                                continue;

                            for (int direction = 0; direction < 6; direction++)
                            {
                                int3 neighborPosition = gridPosition + VoxelUtil.VoxelDirectionOffsets[direction];

                                if (TransparencyCheck(voxels, neighborPosition, chunkSize))
                                    continue;

                                AddQuadByDirection(direction, voxel.data, 1.0f, 1.0f, gridPosition, counter.Increment(), vertices, indices);
                            }
                        }
                    }
                }
            }
        }

        public static bool TransparencyCheck(NativeArray<Voxel> voxels, int3 position, int3 chunkSize)
        {
            if (!VoxelUtil.BoundaryCheck(position, chunkSize))
                return false;

            return voxels[VoxelUtil.To1DIndex(position, chunkSize)].data != Voxel.VoxelType.Air;
        }

        static unsafe void AddQuadByDirection(int direction, Voxel.VoxelType data, float width, float height, int3 gridPosition, int quadIndex, NativeArray<GPUVertex> vertices, NativeArray<int> indices)
        {
            int vertexStart = quadIndex * 4;
            for (int i = 0; i < 4; i++)
            {
                GPUVertex v = new GPUVertex();
                float3 pos = VoxelUtil.CubeVertices[VoxelUtil.CubeFaces[i + direction * 4]];
                pos[VoxelUtil.DirectionAlignedX[direction]] *= width;
                pos[VoxelUtil.DirectionAlignedY[direction]] *= height;
                v.position = pos + gridPosition;
                v.normal = VoxelUtil.VoxelDirectionOffsets[direction];
                int atlasIndex = (int)data * 6 + direction;
                int2 atlasPosition = new int2 { x = atlasIndex % AtlasSize.x, y = atlasIndex / AtlasSize.x };
                v.uv = new float4 { x = VoxelUtil.CubeUVs[i].x * width, y = VoxelUtil.CubeUVs[i].y * height, z = atlasPosition.x, w = atlasPosition.y };
                vertices[vertexStart + i] = v;
            }

            int indexStart = quadIndex * 6;
            int baseVertex = vertexStart;
            for (int i = 0; i < 6; i++)
            {
                indices[indexStart + i] = VoxelUtil.CubeIndices[direction * 6 + i] + baseVertex;
            }
        }
    }
}
