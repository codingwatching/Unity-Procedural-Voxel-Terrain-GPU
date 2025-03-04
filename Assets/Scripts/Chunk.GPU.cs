using System;
using System.Collections;
using OptIn.Voxel;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public partial class Chunk
{
    GPUVoxelData gpuVoxelData;
    GPUMeshData gpuMeshData;

    IEnumerator UpdateGPUMesh()
    {
        if (Updating)
            yield break;
        if (!generator.CanUpdate)
            yield break;

        generator.UpdatingChunks++;

        // 释放旧的 GPUgpuMeshData 并创建新的，预分配大小依据 chunkSize
        gpuMeshData?.Dispose();
        int3 intChunkSize = VoxelUtil.ToInt3(chunkSize);
        gpuMeshData = new GPUMeshData(intChunkSize);
        
        // 调用 GPU 版生成逻辑，将生成部分全部移至 GPUgpuMeshData.Generate 中
        yield return gpuMeshData.Generate(generator.meshComputeShader, voxelData.asyncVoxelBuffer, VoxelUtil.ToInt3(chunkPosition), intChunkSize, VoxelMeshBuilder.AtlasSize);

        // 根据生成面数计算顶点和索引数量
        int faceCount = gpuMeshData.faceCount;
        int vertexCount = faceCount * 4;
        int indexCount = faceCount * 6;

        if (vertexCount > 0 && indexCount > 0)
        {
            // 分别回读顶点和索引数据（必须严格按照 AsyncComputeBuffer 回读流程）
            NativeArray<GPUVertex> vertexNative = new NativeArray<GPUVertex>(vertexCount, Allocator.Persistent);
            gpuMeshData.vertexBuffer.StartReadbackNonAlloc(ref vertexNative, vertexCount);
            while (!gpuMeshData.vertexBuffer.IsDataAvailable())
                yield return null;
            gpuMeshData.vertexBuffer.EndReadback();

            NativeArray<uint> indexNative = new NativeArray<uint>(indexCount, Allocator.Persistent);
            gpuMeshData.indexBuffer.StartReadbackNonAlloc(ref indexNative, indexCount);
            while (!gpuMeshData.indexBuffer.IsDataAvailable())
                yield return null;
            gpuMeshData.indexBuffer.EndReadback();

            mesh.SetVertexBufferParams(vertexCount, cachedVertexAttributes);
            mesh.SetIndexBufferParams(indexCount, IndexFormat.UInt32);

            mesh.SetVertexBufferData(vertexNative, 0, 0, vertexCount, 0, MeshUpdateFlags.DontRecalculateBounds);
            mesh.SetIndexBufferData(indexNative, 0, 0, indexCount, MeshUpdateFlags.DontRecalculateBounds);
            mesh.subMeshCount = 1;
            SubMeshDescriptor desc = new SubMeshDescriptor(0, indexCount, MeshTopology.Triangles);
            mesh.SetSubMesh(0, desc, MeshUpdateFlags.DontRecalculateBounds);
            mesh.RecalculateBounds();

            vertexNative.Dispose();
            indexNative.Dispose();

            if (argent)
                SetSharedMesh(mesh);
            else
                VoxelColliderBuilder.Instance.Enqueue(this, mesh);
        }

        gpuMeshData.Dispose();
        dirty = false;
        argent = false;
        gameObject.layer = LayerMask.NameToLayer("Voxel");
        meshUpdator = null;
        generator.UpdatingChunks--;
        yield break;
    }

    public bool SetGPUVoxel(Vector3Int gridPosition, Voxel.VoxelType type)
    {
        if (!initialized)
        {
            return false;
        }

        if (!VoxelUtil.BoundaryCheck(gridPosition, chunkSize))
        {
            return false;
        }

        int kernel = generator.voxelComputeShader.FindKernel("SetVoxel");
        if (kernel < 0)
        {
            Debug.LogError("SetVoxel kernel not found in voxelComputeShader");
            return false;
        }

        // 绑定用于修改体素数据的 GPU Buffer
        generator.voxelComputeShader.SetBuffer(kernel, "asyncVoxelBuffer", voxelData.asyncVoxelBuffer);

        // 将 gridPosition 与 chunkSize 传入 GPU（用于计算 1D 下标）
        generator.voxelComputeShader.SetInts("gridPosition", gridPosition.x, gridPosition.y, gridPosition.z);
        generator.voxelComputeShader.SetInts("chunkSize", chunkSize.x, chunkSize.y, chunkSize.z);
        generator.voxelComputeShader.SetInt("newVoxelType", (int)type);

        // 这里由于只修改一个体素，所以 dispatch 1 个线程组即可
        generator.voxelComputeShader.Dispatch(kernel, 1, 1, 1);

        dirty = true;
        argent = true;
        return true;
    }
}
