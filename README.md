基于Unity-Procedural-Voxel-Terrain优化的版本 删除所有CPU的Jobs体素生成 改为GPU生成体素然后cpu只管构建网格 在克莱因群分形生成下 加速了10倍以上 让我实现了分形自由
Todo
GPU只回传体素数据 使用体素生成贪婪碰撞体 这样就不需要回传网格了 直接使用Graphics.DrawProceduralIndirect在gpu上渲染区块 

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
