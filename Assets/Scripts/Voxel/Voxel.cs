namespace OptIn.Voxel
{
    public struct Voxel
    {
        // 为实现 Marching Cubes 体素生成，扩展体素数据
        public ushort texId;
        public byte shapeId;
        public byte metadata;

        // 对应 B 中 Vox 的逻辑：
        // 当 shapeId == 0 时认为是 Isosurface
        public float Density
        {
            get { return IsShapeIsosurface ? ((float)metadata - 128f) / 127f : 0f; }
            set
            {
                float clamped = value < -1f ? -1f : (value > 1f ? 1f : value);
                metadata = (byte)(clamped * 127f + 128f);
            }
        }

        public bool IsShapeIsosurface => shapeId == 0;
        public bool IsShapeCube => shapeId == 1;

        public bool IsDensityNil()
        {
            return Density <= 0f;
        }

        public bool IsTexNil()
        {
            return texId <= 0;
        }

        public bool IsAir()
        {
            return IsTexNil() || IsDensityNil();
        }

        public static Voxel Empty => new Voxel { texId = 0, shapeId = 0, metadata = 128 };

        public override string ToString()
        {
            return $"tex: {texId}, shape: {shapeId}, metadata: {metadata}, density: {Density:0.00}";
        }

        public override int GetHashCode()
        {
            return texId ^ shapeId ^ metadata;
        }
    }
}
