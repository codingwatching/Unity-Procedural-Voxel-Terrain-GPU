using System.Collections;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using OptIn.Voxel;
using Tuntenfisch.Generics;

/// <summary>
/// 利用 ComputeShader 在 GPU 上高效生成体素数据，
/// 支持任意形状的 chunkSize，如果尺寸发生变化会动态重新分配缓冲区，
/// 生成数据通过非分配读回+批量内存拷贝返回给 managed 数组。
/// </summary>
public class GPUVoxelData : System.IDisposable
{
    private int3 currentChunkSize; // 当前使用的区块尺寸
    private AsyncComputeBuffer asyncVoxelBuffer;

    public GPUVoxelData(int3 initialChunkSize)
    {
        currentChunkSize = initialChunkSize;
        AllocateBuffer(currentChunkSize);
    }

    /// <summary>
    /// 根据当前区块尺寸重新分配 GPU 缓冲区。
    /// </summary>
    private void AllocateBuffer(int3 size)
    {
        int numVoxels = size.x * size.y * size.z;
        int voxelSize = sizeof(int);
        if (asyncVoxelBuffer != null)
        {
            asyncVoxelBuffer.Release();
        }
        asyncVoxelBuffer = new AsyncComputeBuffer(numVoxels, voxelSize, ComputeBufferType.Default);
    }
    public IEnumerator Generate(Voxel[] voxels, int3 chunkPosition, int3 newChunkSize, ComputeShader computeShader)
    {
        // 若尺寸发生变化，则重新分配 GPU 缓冲区
        if (!newChunkSize.Equals(currentChunkSize))
        {
            currentChunkSize = newChunkSize;
            AllocateBuffer(currentChunkSize);
        }

        int numVoxels = newChunkSize.x * newChunkSize.y * newChunkSize.z;
        int kernel = computeShader.FindKernel("CSMain");
        if (kernel < 0)
        {
            Debug.LogError("未能找到 ComputeShader 内核 CSMain！");
            yield break;
        }

        // 设置 ComputeShader 参数
        computeShader.SetBuffer(kernel, "Result", asyncVoxelBuffer);
        computeShader.SetInts("chunkPosition", chunkPosition.x, chunkPosition.y, chunkPosition.z);
        computeShader.SetInts("chunkSize", newChunkSize.x, newChunkSize.y, newChunkSize.z);

        // 根据新尺寸计算 Dispatch 的线程组数（假设 ComputeShader 中 [numthreads(8,8,8)]）
        int threadGroupX = Mathf.CeilToInt(newChunkSize.x / 8f);
        int threadGroupY = Mathf.CeilToInt(newChunkSize.y / 8f);
        int threadGroupZ = Mathf.CeilToInt(newChunkSize.z / 8f);
        computeShader.Dispatch(kernel, threadGroupX, threadGroupY, threadGroupZ);

        // 创建 NativeArray 接收 GPU 读回数据，使用 Persistent 分配保证内存稳定
        NativeArray<int> nativeData = new NativeArray<int>(numVoxels, Allocator.Persistent);
        asyncVoxelBuffer.StartReadbackNonAlloc(ref nativeData, numVoxels);

        // 等待 GPU 计算及读回完成
        while (!asyncVoxelBuffer.IsDataAvailable())
            yield return null;
        asyncVoxelBuffer.EndReadback();

        // 批量内存拷贝：要求 Voxel 为 blittable（例如仅含一个 int 字段）
        CopyNativeDataToManaged(voxels, nativeData, numVoxels);

        nativeData.Dispose();
        yield break;
    }

    private static unsafe void CopyNativeDataToManaged(Voxel[] voxels, NativeArray<int> nativeData, int numVoxels)
    {
        fixed (Voxel* dest = voxels)
        {
            void* destPtr = dest;
            void* srcPtr = NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(nativeData);
            UnsafeUtility.MemCpy(destPtr, srcPtr, numVoxels * sizeof(Voxel));
        }
    }

    public void Dispose()
    {
        if (asyncVoxelBuffer != null)
        {
            asyncVoxelBuffer.Release();
            asyncVoxelBuffer = null;
        }
    }
}
