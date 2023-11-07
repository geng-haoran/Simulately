# Rigid Body Physics API

### Rigid body & Collision

```python
from pxr import UsdPhysics

# check rigidbody api
prim.HasAPI(UsdPhysics.RigidBodyAPI)
prim.HasAPI(pxr.UsdPhysics.CollisionAPI)

# get api
UsdPhysics.RigidBodyAPI.Get(stage, path)
UsdPhysics.CollisionAPI.Get(stage, path)

# apply/add api

UsdPhysics.RigidBodyAPI.Apply(stage, path)
UsdPhysics.CollisionAPI.Apply(stage, path)
```

### Set up GPU for physics

By default isaac sim physics engine when running standalone will use CPU solver; however, particle physics is only available in GPU mode. TO enable GPU, we need to following lines of code: 

```python
from omni.physx import  acquire_physx_interface

physx = acquire_physx_interface()

physx.overwrite_gpu_setting(1) # 1 means using GPU
```

## set up physics call_back
```python
self._world.add_physics_callback("frankabot_step", callback_fn=self._on_sim_step)
```


Make sure to remove it afterwards

```python
self._world.remove_physics_callback("frankabot_step")
```

### Liquid simulation

There are two ways to simulate liquid
1. Generate liquid on the fly (suitable for faucet-like applications).
2. Pre-generate liquid and let physics handle the final locations of the liquid pool.

For **type-1**: the pipeline is: 

1.1 generate positions for each batch of particles.

1.2 create point instancer and specify its properties.

1.3 create a sphere to temporarily hold the liquid (make it invisible).

1.4 after certain time-steps, we crate one batch. We can do that by creating a callback for on_physics_step

Notice the **activeIndices** attribute, I suspect, we can use this to control what particle to simulate.
More about particle simulation are here: 
[https://graphics.pixar.com/usd/release/api/class_usd_geom_point_instancer.html](https://graphics.pixar.com/usd/release/api/class_usd_geom_point_instancer.html)

```python
def set_up():
        addPhysxParticleSystem(
            stage,
            particleSystemPath,
            contactOffset,
            restOffset,
            particleContactOffset,
            solidRestOffset,
            fluidRestOffset,
            4,
            1,
            Gf.Vec3f(0, 0, 0),
            scenePath
        )

        particleSystem = stage.GetPrimAtPath(particleSystemPath)
        # particle system settings
        particleSystem.GetAttribute("cohesion").Set(0.002)
        particleSystem.GetAttribute("smoothing").Set(0.8)
        particleSystem.GetAttribute("anisotropyScale").Set(1.0)
        particleSystem.GetAttribute("anisotropyMin").Set(0.2)
        particleSystem.GetAttribute("anisotropyMax").Set(2.0)
        particleSystem.GetAttribute("viscosity").Set(0.0091)
        particleSystem.GetAttribute("surfaceTension").Set(0.0074)
        particleSystem.GetAttribute("particleFriction").Set(0.1)
        particleSystem.CreateAttribute("maxParticleNeighborhood", Sdf.ValueTypeNames.Int, True).Set(64)
        particleSystem.GetAttribute("maxParticles").Set(20000)

        # apply isoSurface params
        particleSystem.CreateAttribute("enableIsosurface", Sdf.ValueTypeNames.Bool, True).Set(True)
        particleSystem.CreateAttribute("maxIsosurfaceVertices", Sdf.ValueTypeNames.Int, True).Set(1024 * 1024)
        particleSystem.CreateAttribute("maxIsosurfaceTriangles", Sdf.ValueTypeNames.Int, True).Set(2 * 1024 * 1024)
        particleSystem.CreateAttribute("maxNumIsosurfaceSubgrids", Sdf.ValueTypeNames.Int, True).Set(1024 * 4)
        particleSystem.CreateAttribute("isosurfaceGridSpacing", Sdf.ValueTypeNames.Float, True).Set(0.2)

        filterSmooth = 1

        filtering = 0
        passIndex = 0
        filtering = setGridFilteringPass(filtering, passIndex, filterSmooth)
        passIndex = passIndex + 1
        filtering = setGridFilteringPass(filtering, passIndex, filterSmooth)
        passIndex = passIndex + 1

        particleSystem.CreateAttribute("isosurfaceKernelRadius", Sdf.ValueTypeNames.Float, True).Set(0.5)
        particleSystem.CreateAttribute("isosurfaceLevel", Sdf.ValueTypeNames.Float, True).Set(-0.3)
        particleSystem.CreateAttribute("isosurfaceGridFilteringFlags", Sdf.ValueTypeNames.Int, True).Set(filtering)
        particleSystem.CreateAttribute(
            "isosurfaceGridSmoothingRadiusRelativeToCellSize", Sdf.ValueTypeNames.Float, True
        ).Set(0.3)

        particleSystem.CreateAttribute("isosurfaceEnableAnisotropy", Sdf.ValueTypeNames.Bool, True).Set(False)
        particleSystem.CreateAttribute("isosurfaceAnisotropyMin", Sdf.ValueTypeNames.Float, True).Set(0.1)
        particleSystem.CreateAttribute("isosurfaceAnisotropyMax", Sdf.ValueTypeNames.Float, True).Set(2.0)
        particleSystem.CreateAttribute("isosurfaceAnisotropyRadius", Sdf.ValueTypeNames.Float, True).Set(0.5)

        particleSystem.CreateAttribute("numIsosurfaceMeshSmoothingPasses", Sdf.ValueTypeNames.Int, True).Set(5)
        particleSystem.CreateAttribute("numIsosurfaceMeshNormalSmoothingPasses", Sdf.ValueTypeNames.Int, True).Set(5)

        particleSystem.CreateAttribute("isosurfaceDoNotCastShadows", Sdf.ValueTypeNames.Bool, True).Set(True)

        stage.SetInterpolationType(Usd.InterpolationTypeHeld)

def create_ball():
        # create sphere on points 
        points = self.point_sphere(50, 1)

        # basePos = Gf.Vec3f(11.0, 12.0, 35.0) + pos
        basePos = pos
        positions = [x + basePos for x in points]

        radius = 1
        # particleSpacing = 2.0 * radius * 0.6
        particleSpacing = 4.0 * radius * 0.6

        positions_list = positions
        velocities_list = [Gf.Vec3f(0.0, 0.0, 0.0)] * len(positions)
        protoIndices_list = [0] * len(positions)

        protoIndices = Vt.IntArray(protoIndices_list)
        positions = Vt.Vec3fArray(positions_list)
        velocities = Vt.Vec3fArray(velocities_list)

        particleInstanceStr = "/particlesInstance" + str(self.it)
        particleInstancePath = Sdf.Path(particleInstanceStr)

        # Create point instancer
        pointInstancer = UsdGeom.PointInstancer.Define(stage, particleInstancePath)
        prototypeRel = pointInstancer.GetPrototypesRel()

        # Create particle instance prototypes
        particlePrototype = PhysxParticleInstancePrototype()
        particlePrototype.selfCollision = True
        particlePrototype.fluid = True
        particlePrototype.collisionGroup = 0
        particlePrototype.mass = 0.001

        prototypePath = particleInstancePath.pathString + "/particlePrototype"
        
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(prototypePath))

        sphere.GetRadiusAttr().Set(particleSpacing)
        spherePrim = sphere.GetPrim()
        spherePrim.GetAttribute('visibility').Set('invisible')
        spherePrim.CreateAttribute("enableAnisotropy", Sdf.ValueTypeNames.Bool, True).Set(True)

        particleInstanceApi = PhysxSchema.PhysxParticleAPI.Apply(spherePrim)

        particleInstanceApi.CreateSelfCollisionAttr().Set(particlePrototype.selfCollision)
        particleInstanceApi.CreateFluidAttr().Set(particlePrototype.fluid)
        particleInstanceApi.CreateParticleGroupAttr().Set(particlePrototype.collisionGroup)
        particleInstanceApi.CreateMassAttr().Set(particlePrototype.mass)

        # Reference simulation owner using PhysxPhysicsAPI
        physicsApi = PhysxSchema.PhysxPhysicsAPI.Apply(spherePrim)
        physicsApi.CreateSimulationOwnerRel().SetTargets([self.particleSystemPath])

        # add prototype references to point instancer
        prototypeRel.AddTarget(Sdf.Path(prototypePath))

        # Set active particle indices
        activeIndices = []
        for i in range(len(positions)):
            activeIndices.append(protoIndices[i])

        orientations = [Gf.Quath(1.0, Gf.Vec3h(0.0, 0.0, 0.0))] * len(positions)

        angular_velocities = [Gf.Vec3f(0.0, 0.0, 0.0)] * len(positions)

        pointInstancer.GetProtoIndicesAttr().Set(activeIndices)
        pointInstancer.GetPositionsAttr().Set(positions)
        pointInstancer.GetOrientationsAttr().Set(orientations)
        pointInstancer.GetVelocitiesAttr().Set(velocities)
        pointInstancer.GetAngularVelocitiesAttr().Set(angular_velocities)

```

For **Type-2**,
We just generate the entire shape in advance. 

### Set up physics materials (this is important, we need friction to pick up objects)
```python
    def _setup_physics_material(self, path):
        # def _setup_physics_material(self, path: Sdf.Path):
        from pxr import UsdGeom, UsdLux, Gf, Vt, UsdPhysics, PhysxSchema, Usd, UsdShade, Sdf
        _material_static_friction = 1.0
        _material_dynamic_friction = 1.0
        _material_restitution = 0.0
        _physicsMaterialPath = None

        if _physicsMaterialPath is None:
            _physicsMaterialPath = self._stage.GetDefaultPrim().GetPath().AppendChild("physicsMaterial")
            UsdShade.Material.Define(self._stage, _physicsMaterialPath)
            material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(_physicsMaterialPath))
            material.CreateStaticFrictionAttr().Set(_material_static_friction)
            material.CreateDynamicFrictionAttr().Set(_material_dynamic_friction)
            material.CreateRestitutionAttr().Set(_material_restitution)

        collisionAPI = UsdPhysics.CollisionAPI.Get(self._stage, path)
        prim = self._stage.GetPrimAtPath(path)
        if not collisionAPI:
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        # apply material
        physicsUtils.add_physics_material_to_prim(self._stage, prim, _physicsMaterialPath)
```
### Set mass:
```python
def _apply_mass(self, mesh: UsdGeom.Mesh, mass: float):
    massAPI = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
    massAPI.GetMassAttr().Set(mass)
```


### Set/Get Gravity value

```python
# Assume _my_world is of type World
self._my_world ._physics_context.get_gravity()
meters_per_unit = get_stage_units()
self._my_world ._physics_context.set_gravity(value=-9.81 / meters_per_unit)
```

### Set up prim collision type
```python
from omni.physx.scripts import utils

utils.setCollider(collision_prim, approximationShape="convexDecomposition")
```

### Setup physics materials
we can only pick up stuff if it has frictions.
```python
    def _setup_physics_material(self, path: Sdf.Path):
        from pxr import UsdGeom, UsdLux, Gf, Vt, UsdPhysics, PhysxSchema, Usd, UsdShade, Sdf
        
        if self._physicsMaterialPath is None:
            self._physicsMaterialPath = self._stage.GetDefaultPrim().GetPath().AppendChild("physicsMaterial")
            UsdShade.Material.Define(self._stage, self._physicsMaterialPath)
            material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._physicsMaterialPath))
            material.CreateStaticFrictionAttr().Set(self._material_static_friction)
            material.CreateDynamicFrictionAttr().Set(self._material_dynamic_friction)
            material.CreateRestitutionAttr().Set(self._material_restitution)

        collisionAPI = UsdPhysics.CollisionAPI.Get(self._stage, path)
        prim = self._stage.GetPrimAtPath(path)
        if not collisionAPI:
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        # apply material
        physicsUtils.add_physics_material_to_prim(self._stage, prim, self._physicsMaterialPath)
```

### Set up stiffness and damping
```python
from pxr import UsdPhysics

joint = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
joint.CreateDampingAttr(1e3)
joint.CreateStiffnessAttr(1e3)
```

### Get prim position/rotation/scale
```python
print(prim.GetAttribute("xformOp:orient").Get())
print(prim.GetAttribute("xformOp:translate").Get())
print(prim.GetAttribute("xformOp:scale").Get())
```


# Joint Setup

```python
from pxr import UsdPhysics
```

```python
# type == "SLIDER"
joint = UsdPhysics.PrismaticJoint.Define(stage, path)

# type == "REVOLUTE"
joint = UsdPhysics.RevoluteJoint.Define(self.assembly_stage, p)
```
# Get Joint

```python
UsdPhysics.PrismaticJoint.Get(stage, path)	
```

# Property Setup

```python
                joint.CreateAxisAttr(mate.axis)
                # print(f.limits)
                if mate.limits[0] is not None:
                    joint.CreateLowerLimitAttr(mate.limits[0])
                if mate.limits[1] is not None:
                    joint.CreateUpperLimitAttr(mate.limits[1])

                joint.CreateBody0Rel().SetTargets([base_path])
                joint.CreateBody1Rel().SetTargets([prim_path])

                joint.CreateLocalPos0Attr().Set((joint_global_pose * body_0_global.GetInverse()).ExtractTranslation())
                joint.CreateLocalRot0Attr().Set(
                    Gf.Quatf((joint_global_pose * body_0_global.GetInverse()).ExtractRotation().GetQuat())
                )

                joint.CreateLocalPos1Attr().Set((joint_global_pose * body_1_global.GetInverse()).ExtractTranslation())
                joint.CreateLocalRot1Attr().Set(
                    Gf.Quatf((joint_global_pose * body_1_global.GetInverse()).ExtractRotation().GetQuat())
                )
```
