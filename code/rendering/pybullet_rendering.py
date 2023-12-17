#make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
#otherwise use testrender.py (slower but compatible without numpy)
#you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)

import numpy as np
from PIL import Image
import pybullet
import time
import pybullet_data

# plt.ion()

img = np.random.rand(480, 640)


pybullet.connect(pybullet.GUI)
# pybullet.connect(pybullet.DIRECT)

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.loadURDF("plane.urdf", [0, 0, 0])

box_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
box_visual_id = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
box_pose = [0, 0, 0.5]  # Position the box slightly above the ground
box_body_id = pybullet.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_id, baseVisualShapeIndex=box_visual_id, basePosition=box_pose)

camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]
pybullet.setGravity(0, 0, -10)

SAVE_IMG = False

pitch = np.arctan2(2, 4) * 180 / np.pi
roll = 0
yaw = 0

upAxisIndex = 2
camDistance = -np.sqrt(20)
pixelWidth = 640
pixelHeight = 480
nearPlane = 0.05
farPlane = 100

fov = 1 / np.pi * 180

main_start = time.time()
render_times = []
render_times1 = []
for i in range(1000):
  pybullet.stepSimulation()

  viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                          roll, upAxisIndex)
  aspect = pixelWidth / pixelHeight
  projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
  start = time.time()
  img_arr = pybullet.getCameraImage(pixelWidth,
                                    pixelHeight,
                                    viewMatrix,
                                    projectionMatrix,
                                    shadow=1,
                                    lightDirection=[1, 1, 1],
                                    renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                                    flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
  stop = time.time()
  print("renderImage %f" % (stop - start))
  render_times.append(stop - start)
  
  w = img_arr[0]  #width of the image, in pixels
  h = img_arr[1]  #height of the image, in pixels
  rgb = img_arr[2]  #color data RGB
  dep = img_arr[3]  #depth data
  seg = img_arr[4]  #segmentation data


  #note that sending the data to matplotlib is really slow

  #reshape is needed
  if SAVE_IMG:
    np_img_arr = np.reshape(rgb, (h, w, 4))
    np_img_arr = np_img_arr
    img = Image.fromarray(np_img_arr)
    img.save('test.png')
  

main_stop = time.time()

print("Total time %f" % (main_stop - main_start))
render_times_arr = np.array(render_times)
print("Mean render time: ", np.mean(render_times_arr))

pybullet.resetSimulation()
