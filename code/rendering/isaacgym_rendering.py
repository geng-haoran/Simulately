import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize Isaac Gym
gym_instance = gym.make('Humanoid-v0')

# Set up the scene
gym_instance.reset()
gym_instance.render(mode='human')

# Render a single image
image = gym_instance.render(mode='rgb_array')

# Save the image using matplotlib
plt.imshow(image)
plt.savefig('/mnt/data/isaacgym_rendered_image.png')
