# Unity 程序化体素地形

## 概述
这是 **Unity-Procedural-Voxel-Terrain** 的优化版本，完全 **移除了 CPU Jobs 体素生成**，改为 **GPU 计算生成体素**，CPU 仅负责 **网格构建** 和prefab生成

删除了GPU的**网格构建** 因为实际还是要回传创建碰撞体 而且做游戏的情况下需要更好的自定义网格因此放弃全部过早优化设计。

chunk会比实际chunk大一圈 比如32实际上是34 这是为了解决不同区块的不连贯生成问题 并且保证局部性故意不去访问临近区块以实现区块局部性从而更好的在未来支持ECS
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
