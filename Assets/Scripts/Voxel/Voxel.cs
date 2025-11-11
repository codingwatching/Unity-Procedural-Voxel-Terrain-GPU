using Unity.Mathematics;
using UnityEngine;

namespace OptIn.Voxel
{
    [System.Serializable]
    public struct Voxel
    {
        /// <summary>
        /// ID of the voxel.
        /// > 0: Block type ID.
        /// < 0: Isosurface material ID (absolute value).
        /// = 0: Air.
        /// </summary>
        public int voxelID;

        /// <summary>
        /// Metadata byte. For isosurface voxels (voxelID <= 0), this stores density.
        /// </summary>
        public byte metadata;

        public static Voxel Empty => new Voxel { voxelID = 0, metadata = 0 };

        public float Density
        {
            get
            {
                if (voxelID > 0) return 1f; // Blocks are always "full"
                return (metadata - 128) / 127f;
            }
            set
            {
                metadata = (byte)(math.clamp(value, -1f, 1f) * 127f + 128f);
            }
        }

        public bool IsBlock => voxelID > 0;
        public bool IsIsosurface => voxelID <= 0; // Air is part of the isosurface field
        public bool IsAir => voxelID == 0 && Density <= 0;
        public bool IsSolid => IsBlock || Density > 0;

        public ushort GetMaterialID()
        {
            // Returns a positive ID for texturing, regardless of type.
            return (ushort)Mathf.Abs(voxelID);
        }
    }
}