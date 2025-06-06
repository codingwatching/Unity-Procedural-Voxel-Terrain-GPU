# Unity 程序化体素地形（优化版）

## 概述
这是 **Unity-Procedural-Voxel-Terrain** 的优化版本，完全 **移除了 CPU Jobs 体素生成**，改为 **GPU 计算生成体素**，CPU 仅负责 **网格构建** 也提供了GPU的**网格构建** 但在我的笔记本上得到了更低的帧数 原因不明所以默认关闭了。在 **克莱因群分形生成** 下，性能提升 **10 倍以上**，实现了真正的分形自由。

## Todo
- **GPU 只回传体素数据，使用体素生成贪婪碰撞体，从而无需回传网格，并直接使用 `Graphics.DrawProceduralIndirect` 在 GPU 上渲染区块**

# Unity-Procedural-Voxel-Terrain

Procedural Voxel Terrain Project

![Main](./images/main.png)

# Mesh Optimization

## Stupid Method

![Stupid](./images/stupid.png)

## Culling Method

![Stupid](./images/culling.png)

## Greedy Meshing Only Y Axis

![Stupid](./images/greedy_meshing_only_y.png)

## Greedy Meshing

![Stupid](./images/greedy_meshing.png)

# Texture Mapping

![Texture Mapping](./images/texturing.png)

# Ambient Occlusion

![Ambient Occlusion](./images/ambient_occlusion.png)

## With Greedy Meshing

![Ambient Occlusion with Greedy Meshing](./images/ambient_occlusion_with_greedy_meshing.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
