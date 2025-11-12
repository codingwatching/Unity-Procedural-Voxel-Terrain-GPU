using OptIn.Voxel;
using UnityEngine;

public class VoxelController : MonoBehaviour
{
    [Header("Block Editing")]
    [Tooltip("The ID for blocks placed with the left mouse button.")]
    [SerializeField] short blockMaterialId = 3; // Stone

    [Header("Smooth Voxel Editing")]
    [SerializeField] float smoothEditRadius = 3f;
    [SerializeField] float smoothEditIntensity = 0.5f;
    [Tooltip("The material ID (as a negative number) for smooth voxels.")]
    [SerializeField] short smoothMaterialId = -3; // Stone material for isosurface

    void Update()
    {
        var terrainGenerator = TerrainGenerator.Instance;
        if (terrainGenerator == null) return;

        HandleBlockEditing(terrainGenerator);
    }

    private void HandleBlockEditing(TerrainGenerator generator)
    {
        // Place Block
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
            if (Physics.Raycast(ray, out RaycastHit hit, 100f))
            {
                Vector3 placePosition = hit.point - ray.direction * 0.01f;
                // VoxelID > 0 for blocks
                generator.SetVoxel(placePosition, blockMaterialId);
            }
        }

        // Remove Block
        if (Input.GetMouseButtonDown(1))
        {
            Ray ray = Camera.main.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
            if (Physics.Raycast(ray, out RaycastHit hit, 100f))
            {
                Vector3 removePosition = hit.point + ray.direction * 0.01f;
                generator.SetVoxel(removePosition, 0); // VoxelID 0 is Air
            }
        }
    }

    private void HandleSmoothEditing(TerrainGenerator generator)
    {
        // Add density (build)
        if (Input.GetMouseButton(0))
        {
            Ray ray = Camera.main.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
            if (Physics.Raycast(ray, out RaycastHit hit, 100f))
            {
                // VoxelID < 0 for isosurface materials
                generator.ModifySphere(hit.point, smoothEditRadius, smoothEditIntensity, smoothMaterialId);
            }
        }

        // Subtract density (carve)
        if (Input.GetMouseButton(1))
        {
            Ray ray = Camera.main.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
            if (Physics.Raycast(ray, out RaycastHit hit, 100f))
            {
                generator.ModifySphere(hit.point, smoothEditRadius, -smoothEditIntensity, 0);
            }
        }
    }
}