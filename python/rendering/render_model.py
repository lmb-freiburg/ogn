import math
import numpy as np
import bmesh
import bpy
import sys
from rendering import python_octree

MAX_NUM_LEVELS = 5

CLASS_EMPTY = 0
CLASS_FILLED = 1

cube_materials = []
proto_cubes = []

def clear_selection():
  bpy.context.scene.objects.active = None
  for o in bpy.data.objects:
    o.select = False

def select_object(obj):
  clear_selection()
  bpy.context.selected_objects.clear()
  bpy.context.scene.objects.active = obj
  obj.select = True
  return obj

def setup_camera():
    # set position
    scene = bpy.data.scenes["Scene"]
    scene.camera.location.x = 0
    scene.camera.location.y = 0
    scene.camera.location.z = 0
    bpy.data.cameras["Camera"].clip_end = 10000.

    # track invisible cube at (0, 0, 0)
    bpy.data.objects['Cube'].hide_render = True
    ttc = scene.camera.constraints.new(type='TRACK_TO')
    ttc.target = bpy.data.objects['Cube']
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis = 'UP_Y'

def setup_proto_cubes():
    for level in range(0, MAX_NUM_LEVELS):
      mat = bpy.data.materials.new('cube_material_%02d'%(level))
      col = float(level + 1) / MAX_NUM_LEVELS / 3
      mat.diffuse_color = (col,col,col)
      cube_materials.append(mat)
      bpy.ops.mesh.primitive_cube_add(location=(-1000,-1000,-1000),
                                      rotation=(0,0,0),
                                      radius=1-.00625)
      cube = bpy.context.object
      select_object(cube)
      cube.name = 'proto_cube_level_%01d'%(level)
      bpy.ops.object.material_slot_add()
      cube.material_slots[0].material = mat
      bpy.ops.object.modifier_add(type='BEVEL')
      cube.modifiers['Bevel'].width = 0.2
      cube.modifiers['Bevel'].segments = 3
      proto_cubes.append(cube)

def setup_general():
    bpy.data.worlds["World"].horizon_color = (1, 1, 1)
    bpy.data.scenes["Scene"].render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.samples = 72
    bpy.data.scenes["Scene"].use_nodes = True
    bpy.data.scenes["Scene"].render.filepath = "animation/"
    bpy.data.scenes["Scene"].render.use_compositing = False
    bpy.data.scenes["Scene"].render.layers["RenderLayer"].use_pass_z = False
    bpy.data.screens['Default'].scene = bpy.data.scenes['Scene']

def load_data(file_name):
    vg, resolution = python_octree.import_ot(file_name)
    global_scale = 128.0 / resolution

    ply_vertices = {}
    for key in vg:
        if np.uint32(vg[key]) == CLASS_FILLED:
            x, y, z, side_len = python_octree.get_cube_params(key, resolution)
            x -= resolution//2
            y -= resolution//2
            z -= resolution//2
            if side_len not in ply_vertices:
              ply_vertices[side_len] = []
            ply_vertices[side_len].append('%f %f %f'%(x*global_scale,y*global_scale,z*global_scale))

    cnt = 1

    ply_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
end_header
%s"""

    for side_len in ply_vertices:
      print(side_len)
      path = 'tmp-ply-%d.ply'%(side_len)
      with open(path, 'w') as f:
        f.write(ply_template%(len(ply_vertices[side_len]),
                              '\n'.join(ply_vertices[side_len])))
      bpy.ops.import_mesh.ply(filepath=path)
      obj = bpy.context.selected_objects[0]
      bpy.ops.object.particle_system_add()
      ps = obj.particle_systems[0]
      ps = bpy.data.particles[-1]
      ps.count = len(ply_vertices[side_len])
      ps.emit_from = 'VERT'
      ps.use_emit_random = False
      ps.normal_factor = 0.0
      ps.physics_type = 'NO'
      ps.use_render_emitter = False
      ps.show_unborn = True
      ps.use_dead = False
      ps.lifetime = 250
      ps.render_type = 'OBJECT'
      ps.dupli_object = proto_cubes[-cnt]
      ps.particle_size = 0.5*side_len*global_scale

      cnt += 1

def parse_arguments(argv):
    cnt = 0
    for arg in argv:
        if arg == "-P":
            return argv[cnt+2]
        cnt += 1

def main():
  file_name = parse_arguments(sys.argv)
  setup_camera()
  setup_proto_cubes()
  load_data(file_name)
  setup_general()

if __name__ == '__main__':
  main()
