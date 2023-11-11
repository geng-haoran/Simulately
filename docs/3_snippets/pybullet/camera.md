---
sidebar_position: 2
---

# Camera Sensors

### Create a camera and render an image

```python
import pybullet as p
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
print("plugin=", plugin)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf", [0, 0, -1])
p.loadURDF("r2d2.urdf")

pixelWidth = 320
pixelHeight = 220
camTargetPos = [0, 0, 0]
camDistance = 4
pitch = -10.0
roll = 0
upAxisIndex = 2

while (p.isConnected()):
  for yaw in range(0, 360, 10):
    start = time.time()
    p.stepSimulation()
    stop = time.time()
    print("stepSimulation %f" % (stop - start))

    #viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
                                                     upAxisIndex)
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
    ]

    start = time.time()
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1])
    stop = time.time()
    print("renderImage %f" % (stop - start))
    #time.sleep(.1)
    #print("img_arr=",img_arr)

p.unloadPlugin(plugin)
```

### Render Image Depth and Segmentation
```python

import matplotlib.pyplot as plt
import pybullet
import pybullet as p
import time
import numpy as np  #to reshape for matplotlib

plt.ion()

img = [[1, 2, 3] * 50] * 100  #np.random.rand(200, 320)
#img = [tandard_normal((50,100))
image = plt.imshow(img, interpolation='none', animated=True, label="blah")
ax = plt.gca()
import pybullet_data

pybullet.connect(pybullet.DIRECT)

if 1:
  import pkgutil
  egl = pkgutil.get_loader('eglRenderer')
  if (egl):
    pluginId = pybullet.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
  else:
    pluginId = pybullet.loadPlugin("eglRendererPlugin")
  print("pluginId=",pluginId)

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

#pybullet.loadPlugin("eglRendererPlugin")
usePlane = False
updateHeightfield = True
textureId = -1
if usePlane:
  pybullet.loadURDF("plane.urdf", [0, 0, -1])
else:
  

  useProgrammatic = 0
  useTerrainFromPNG = 1
  useDeepLocoCSV = 2
  

  heightfieldSource = useProgrammatic
  import random
  random.seed(10)
  #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  heightPerturbationRange = 0.05
  if heightfieldSource==useProgrammatic:
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0,heightPerturbationRange)
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
        
        
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[0.05,0.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    #terrainShape = p.createCollisionShape(shapeType = p.GEOM_SPHERE, radius=1)
    terrain  = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrain,[0,0,-1], [0,0,0,1])

  if heightfieldSource==useDeepLocoCSV:
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.5,.5,2.5],fileName = "heightmaps/ground0.txt", heightfieldTextureScaling=128)
    terrain  = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])

  if heightfieldSource==useTerrainFromPNG:
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.1,.1,24],fileName = "heightmaps/wm_height_out.png")
    textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
    terrain  = p.createMultiBody(0, terrainShape)
    p.changeVisualShape(terrain, -1, textureUniqueId = textureId)
   
   
  p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])
#pybullet.loadURDF("r2d2.urdf")

pybullet.setGravity(0, 0, -10)
camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]

pitch = -10.0

roll = 0
upAxisIndex = 2
camDistance = 4
pixelWidth = 320
pixelHeight = 200
nearPlane = 0.01
farPlane = 100

fov = 60

main_start = time.time()
while 1:
  for yaw in range(0, 360, 10):
    if updateHeightfield and heightfieldSource==useProgrammatic:
      for j in range (int(numHeightfieldColumns/2)):
        for i in range (int(numHeightfieldRows/2) ):
          height = random.uniform(0,heightPerturbationRange)#+math.sin(time.time())
          heightfieldData[2*i+2*j*numHeightfieldRows]=height
          heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
          heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
          heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
      #GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
      #GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
      #flags = p.GEOM_CONCAVE_INTERNAL_EDGE
      flags = 0
      print("update!")
      terrainShape2 = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, flags = flags, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, replaceHeightfieldIndex = terrainShape)
      
    pybullet.stepSimulation()
    start = time.time()
    viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                            roll, upAxisIndex)
    aspect = pixelWidth / pixelHeight
    projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = pybullet.getCameraImage(pixelWidth,
                                      pixelHeight,
                                      viewMatrix,
                                      projectionMatrix,
                                      shadow=1,
                                      lightDirection=[1, 1, 1],
                                      renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    stop = time.time()
    print("renderImage %f" % (stop - start))

    w = img_arr[0]  #width of the image, in pixels
    h = img_arr[1]  #height of the image, in pixels
    rgb = img_arr[2]  #color data RGB
    dep = img_arr[3]  #depth data
    #print(rgb)
    print('width = %d height = %d' % (w, h))

    #note that sending the data using imshow to matplotlib is really slow, so we use set_data

    #plt.imshow(rgb,interpolation='none')

    #reshape is needed
    np_img_arr = np.reshape(rgb, (h, w, 4))
    np_img_arr = np_img_arr * (1. / 255.)

    image.set_data(np_img_arr)
    ax.plot([0])
    #plt.draw()
    #plt.show()
    plt.pause(0.01)
    

main_stop = time.time()

print("Total time %f" % (main_stop - main_start))

pybullet.resetSimulation()
```