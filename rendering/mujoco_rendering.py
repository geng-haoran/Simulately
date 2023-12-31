import time

import mujoco
# import mujoco.viewer

from PIL import Image
import numpy as np

hellworld = r"""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="1 1 1 1"/>
    <body pos="0 0 0.1 ">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

#m = mujoco.MjModel.from_xml_path('hello.xml')
m = mujoco.MjModel.from_xml_string(hellworld)
d = mujoco.MjData(m)

renderer = mujoco.Renderer(m, 480, 640)

mujoco.mj_forward(m, d)
renderer.update_scene(d)

N = 2000
frames = []

test_mode = "RGB"
# test_mode = "DEPTH"
# test_mode = "SEG"

if test_mode == "DEPTH": 
    renderer.enable_depth_rendering()
elif test_mode == "SEG":
    renderer.enable_segmentation_rendering()

write_to_file = False

s = time.time()
for i in range(N):
    mujoco.mj_step(m, d)

    renderer.update_scene(d)
    out = renderer.render()
    
    if write_to_file and test_mode == "RGB":
      Image.fromarray(out).save("rgb.png")
    
    if write_to_file and test_mode == "DEPTH":
      # Shift nearest values to the origin.
      out -= out.min()
      # Scale by 2 mean distances of near rays.
      out /= 2*out[out <= 1].mean()
      out = (255*np.clip(out, 0, 1)).astype(np.uint8)
      # depths.append(out)
      Image.fromarray(out).save("depth.png")

    out = renderer.render()
    if write_to_file and test_mode == "SEG":
      geom_ids = out[:, :, 0]
      geom_ids = geom_ids.astype(np.float64) + 1
      geom_ids = geom_ids / geom_ids.max()
      out = (255*geom_ids).astype(np.uint8)
      Image.fromarray(out).save("seg.png")

    frames.append(out)
    
e = time.time()
print(f"{test_mode} FPS: ", N/(e-s))

""" RESULTS on (RTX 2080TI)
  RGB FPS:  1006.2770196925943
  DEPTH FPS:  396.12069736933626
  SEG FPS:  96.72803766492979
"""

""" RESULTS on (4090)
SEG FPS:  165.23122106398043
DEPTH FPS:  563.3816513824315
RGB FPS:  1381.6397861018231
"""