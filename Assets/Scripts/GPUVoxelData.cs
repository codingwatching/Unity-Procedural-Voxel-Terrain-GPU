using System.Collections;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using OptIn.Voxel;
using Tuntenfisch.Generics;
using Tuntenfisch.Extensions;

/// <summary>
/// 利用 ComputeShader 在 GPU 上高效生成体素数据，
/// 支持任意形状的 chunkSize，如果尺寸发生变化会动态重新分配缓冲区，
/// 生成数据通过非分配读回+批量内存拷贝返回给 managed 数组。
/// </summary>
public class GPUVoxelData : System.IDisposable
{
    private int3 currentChunkSize; // 当前使用的区块尺寸
    private AsyncComputeBuffer _asyncVoxelBuffer;
    public AsyncComputeBuffer asyncVoxelBuffer => _asyncVoxelBuffer;

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
        if (_asyncVoxelBuffer != null)
        {
            _asyncVoxelBuffer.Release();
        }
        _asyncVoxelBuffer = new AsyncComputeBuffer(numVoxels, voxelSize, ComputeBufferType.Default);
    }
    public IEnumerator Generate(Voxel[] voxels, int3 chunkPosition, int3 newChunkSize, ComputeShader computeShader, VoxelMeshBuilder.SimplifyingMethod simplifyingMethod)
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
        computeShader.SetBuffer(kernel, "asyncVoxelBuffer", _asyncVoxelBuffer);
        computeShader.SetInts("chunkPosition", chunkPosition.x, chunkPosition.y, chunkPosition.z);
        computeShader.SetInts("chunkSize", newChunkSize.x, newChunkSize.y, newChunkSize.z);

        computeShader.Dispatch(kernel, newChunkSize);
        
        if (simplifyingMethod == VoxelMeshBuilder.SimplifyingMethod.GPUCulling) yield break;

        // 创建 NativeArray 接收 GPU 读回数据，使用 Persistent 分配保证内存稳定
        NativeArray<int> nativeData = new NativeArray<int>(numVoxels, Allocator.Persistent);
        _asyncVoxelBuffer.StartReadbackNonAlloc(ref nativeData, numVoxels);

        // 等待 GPU 计算及读回完成
        while (!_asyncVoxelBuffer.IsDataAvailable())
            yield return null;
        _asyncVoxelBuffer.EndReadback();

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
        if (_asyncVoxelBuffer != null)
        {
            _asyncVoxelBuffer.Release();
            _asyncVoxelBuffer = null;
        }
    }
}
