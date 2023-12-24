"""Hello world for IsaacGym.

Concepts:
    - Engine and scene
    - Renderer, viewer, lighting
    - Run a simulation loop

Notes:
    - For one process, you can only create one engine and one renderer.
"""

from isaacgym import gymapi, gymutil
import os

def main():
    gym = gymapi.acquire_gym()  # Create the GymAPI Instance

    ## get default set of parameters
    sim_params = gymapi.SimParams()

    ## set common parameters
    sim_params.dt = 1 / 60
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    ## configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up axis
    # plane_params.normal = gymapi.Vec3(0, 1, 0) # y-up axis
    plane_params.distance = 0 # specify where the ground plane be placed

    ## create the ground plane
    gym.add_ground(sim, plane_params)

    # Create an Environment Space
    env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
    env_upper = gymapi.Vec3(2.0, 2.0, 0.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Load an Asset
    asset_root = os.path.dirname(os.path.abspath(__file__))
    asset_file = "nv_ant.xml"
    ## load asset with default control type of position for all joints
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    ant_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # Load an Actor
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
    initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    ant_actor = gym.create_actor(env, ant_asset, initial_pose, 'nv_ant')

    # start simulating and rendering
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # step the rendering of physics results
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # sync_frame_time throttle down the simulation rate to real time
        gym.sync_frame_time(sim)

if __name__ == '__main__':
    main()
