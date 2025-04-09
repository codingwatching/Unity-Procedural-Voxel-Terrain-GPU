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

            public NativeMeshData(int3 chunkSize)
            {
                int numVoxels = chunkSize.x * chunkSize.y * chunkSize.z;
                nativeVoxels = new NativeArray<Voxel>(numVoxels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                nativeVertices = new NativeArray<GPUVertex>(0, Allocator.Persistent);
                nativeIndices = new NativeArray<int>(0, Allocator.Persistent);
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
            }

            public IEnumerator ScheduleMeshingJob(Voxel[] voxels, int3 chunkSize, SimplifyingMethod method, bool argent = false)
            {
                nativeVoxels.CopyFrom(voxels);
                VertexBuffer vbuf = new VertexBuffer();
                MarchingCubesGenerateMesh(vbuf, nativeVoxels, chunkSize);

                int vCount = vbuf.Vertices.Count;
                int iCount = vbuf.Indices.Count;
                if (nativeVertices.IsCreated) nativeVertices.Dispose();
                nativeVertices = new NativeArray<GPUVertex>(vCount, Allocator.Persistent);
                if (nativeIndices.IsCreated) nativeIndices.Dispose();
                nativeIndices = new NativeArray<int>(iCount, Allocator.Persistent);

                for (int i = 0; i < vCount; i++)
                {
                    var v = vbuf.Vertices[i];
                    nativeVertices[i] = new GPUVertex
                    {
                        position = v.pos,
                        normal = v.norm,
                        uv = new Vector4(v.uv.x, v.uv.y, 0f, 0f)
                    };
                }
                for (int i = 0; i < iCount; i++)
                {
                    nativeIndices[i] = vbuf.Indices[i];
                }

                yield return null;
            }

            public void GetMeshInformation(out int verticeSize, out int indicesSize)
            {
                verticeSize = nativeVertices.Length;
                indicesSize = nativeIndices.Length;
            }
        }

        // Marching Cubes 体素生成实现，提取自 B 中核心算法
        public static void MarchingCubesGenerateMesh(VertexBuffer vbuf, NativeArray<Voxel> voxels, int3 chunkSize)
        {
            int3[] AXES = new int3[3]
            {
                new int3(1, 0, 0),
                new int3(0, 1, 0),
                new int3(0, 0, 1)
            };
            int3[,] ADJACENT = new int3[3, 6]
            {
                { new int3(0, 0, 0), new int3(0, -1, 0), new int3(0, -1, -1), new int3(0, -1, -1), new int3(0, 0, -1), new int3(0, 0, 0) },
                { new int3(0, 0, 0), new int3(0, 0, -1), new int3(-1, 0, -1), new int3(-1, 0, -1), new int3(-1, 0, 0), new int3(0, 0, 0) },
                { new int3(0, 0, 0), new int3(-1, 0, 0), new int3(-1, -1, 0), new int3(-1, -1, 0), new int3(0, -1, 0), new int3(0, 0, 0) }
            };
            int3[] SN_VERT = new int3[8]
            {
                new int3(0,0,0), new int3(0,0,1), new int3(0,1,0), new int3(0,1,1),
                new int3(1,0,0), new int3(1,0,1), new int3(1,1,0), new int3(1,1,1)
            };
            int[,] EDGE = new int[12, 2]
            {
                { 0, 4 },
                { 1, 5 },
                { 2, 6 },
                { 3, 7 },
                { 5, 7 },
                { 1, 3 },
                { 4, 6 },
                { 0, 2 },
                { 4, 5 },
                { 0, 1 },
                { 6, 7 },
                { 2, 3 }
            };

            for (int i = 0; i < chunkSize.x; i++)
            {
                for (int j = 0; j < chunkSize.y; j++)
                {
                    for (int k = 0; k < chunkSize.z; k++)
                    {
                        int3 pos = new int3(i, j, k);
                        Voxel c = GetVoxel(voxels, pos, chunkSize);
                        for (int l = 0; l < 3; l++)
                        {
                            int3 neighborPos = pos + AXES[l];
                            Voxel neighbor = GetVoxel(voxels, neighborPos, chunkSize);
                            if (!SignChanged(c, neighbor))
                                continue;
                            for (int m = 0; m < 6; m++)
                            {
                                int faceIndex = (!IsDensityValid(c)) ? (5 - m) : m;
                                int3 cellPos = pos + ADJACENT[l, faceIndex];
                                FeatureResult fr = ComputeFeaturePointAndNormal(voxels, cellPos, chunkSize, EDGE, SN_VERT);
                                float3 fp = fr.featurePoint;
                                if (!IsFinite(fp))
                                    fp = new float3(0f, -99f, 0f);
                                float3 vertexPos = cellPos + fp;
                                int texId = DetermineTexId(voxels, cellPos, chunkSize, SN_VERT);
                                vbuf.PushVertex(vertexPos, new UnityEngine.Vector2(texId, -1f), fr.normal);
                            }
                        }
                    }
                }
            }

            // 【修改】如果未手动构造 index 列表，则根据顶点顺序自动生成索引
            if (vbuf.Indices.Count == 0)
            {
                for (int idx = 0; idx < vbuf.Vertices.Count; idx++)
                {
                    vbuf.Indices.Add(idx);
                }
            }
        }

        // 辅助方法
        static Voxel GetVoxel(NativeArray<Voxel> voxels, int3 pos, int3 chunkSize)
        {
            if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= chunkSize.x || pos.y >= chunkSize.y || pos.z >= chunkSize.z)
                return Voxel.Empty;
            int index = pos.z + pos.y * chunkSize.z + pos.x * chunkSize.y * chunkSize.z;
            return voxels[index];
        }

        static bool SignChanged(Voxel a, Voxel b)
        {
            return (a.Density > 0f) != (b.Density > 0f);
        }

        static bool IsDensityValid(Voxel v)
        {
            return !v.IsDensityNil();
        }

        static bool IsFinite(float3 v)
        {
            return math.all(math.isfinite(v));
        }

        struct FeatureResult
        {
            public float3 featurePoint;
            public float3 normal;
        }

        static FeatureResult ComputeFeaturePointAndNormal(NativeArray<Voxel> voxels, int3 pos, int3 chunkSize, int[,] EDGE, int3[] SN_VERT)
        {
            float gx = GetVoxel(voxels, pos + new int3(1, 0, 0), chunkSize).Density - GetVoxel(voxels, pos - new int3(1, 0, 0), chunkSize).Density;
            float gy = GetVoxel(voxels, pos + new int3(0, 1, 0), chunkSize).Density - GetVoxel(voxels, pos - new int3(0, 1, 0), chunkSize).Density;
            float gz = GetVoxel(voxels, pos + new int3(0, 0, 1), chunkSize).Density - GetVoxel(voxels, pos - new int3(0, 0, 1), chunkSize).Density;
            float3 grad = new float3(gx, gy, gz) / 2f;
            float3 normal = math.normalize(-grad);

            float3 sum = new float3(0f, 0f, 0f);
            int count = 0;
            for (int i = 0; i < 12; i++)
            {
                int3 v0 = SN_VERT[EDGE[i, 0]];
                int3 v1 = SN_VERT[EDGE[i, 1]];
                Voxel c0 = GetVoxel(voxels, pos + v0, chunkSize);
                Voxel c1 = GetVoxel(voxels, pos + v1, chunkSize);
                if (SignChanged(c0, c1))
                {
                    float t = math.unlerp(c0.Density, c1.Density, 0f);
                    float3 fp = math.lerp(v0, v1, t);
                    sum += fp;
                    count++;
                }
            }
            float3 featurePoint = (count > 0) ? (sum / count) : new float3(0f, 0f, 0f);
            return new FeatureResult { featurePoint = featurePoint, normal = normal };
        }

        static int DetermineTexId(NativeArray<Voxel> voxels, int3 pos, int3 chunkSize, int3[] SN_VERT)
        {
            float minDensity = float.PositiveInfinity;
            int selectedTex = 0;
            for (int i = 0; i < SN_VERT.Length; i++)
            {
                Voxel v = GetVoxel(voxels, pos + SN_VERT[i], chunkSize);
                if (!v.IsTexNil() && !v.IsDensityNil() && v.Density < minDensity)
                {
                    minDensity = v.Density;
                    selectedTex = v.texId;
                }
            }
            return selectedTex;
        }
    }
}
