# Explanation of the code

## Linear feedback

In the file `run_linear_basic.py`, we just apply the action with a gain and the observations.

## Utils

In the `utils.py` file, the function compute_msfos is used to compute by dichotomy the msfos with respect to the algorithm we wrote in the report :

```plaintext
1. Initialize lower_bound, upper_bound, tolerance, num_trials
2. While (upper_bound - lower_bound > tolerance):
     a. mid_force = (lower_bound + upper_bound) / 2
     b. success_probability = EstimateSuccessProbability(mid_force, num_trials)
     c. If (success_probability > 2/3):
            lower_bound = mid_force
        Else:
            upper_bound = mid_force
3. Calculate MSFOS = (lower_bound + upper_bound) / 2
4. Return MSFOS
```

## New Wrapper

The file `train.py` that is in the ppo_balancer by default has no application of forces to the torso. Therefore, we need to add a new wrapper to the environment, it is in the `train.py` in this repertory.

```python
class VelocityEnvWrapper(gymnasium.Wrapper):
    """
    A custom wrapper for the velocity environment.

    This wrapper allows us to modify or log observations, rewards,
    or interactions with the environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.counttotal = 0
        self.truncated = 0
        self.max_steps = 3000
        self.force_schedule = []
        


    def reset(self, **kwargs):
        """
        Reset the environment and modify the initial observation if needed.
        """
        if not self.truncated:
            print("truncated ! ")
            print(self.force_schedule)
        else: print("resisted",self.force_schedule)
        obs, info = self.env.reset(**kwargs)
        self.count = 0
        self.scales = [0, 5, 0, 9, 0, 12, 0]  # Corresponding scales for the random values
        self.num_intervals = 3
        
        
        # boundaries = sorted(np.random.choice(range(1, self.max_steps), self.num_intervals - 1, replace=False))
        boundaries = [0,200,400,800, 1000, 1700, 1900, self.max_steps]


        # Generate the force schedule with random steps
        self.force_schedule = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            scale = self.scales[min(i, len(self.scales) - 1)]
            force = [np.clip(np.random.normal(loc=scale/3, scale=scale/2),-scale,scale), 0.0, 0.0]
            self.force_schedule.append((start, end, force))
        return obs, info
    
    def apply_external_force(self,force):
        bullet_action = {
            "external_forces": {
                "torso": {
                    "force": force,
                    "local": False,
                }
            }
        }
        self.env.bullet_extra(bullet_action)

    def step(self, action):
        """
        Execute a step in the environment and modify the results if needed.
        """
        
        force = self._get_force_for_step()

        # Apply the determined force to the robot
        self.apply_external_force(force)

        obs, reward, done, truncated, info = self.env.step(action)

        # reward = self.modify_reward(reward)

        self.count += 1
        self.counttotal+=1
        self.truncated = done
        return obs, reward, done, truncated, info
    
    def _get_force_for_step(self):
        """
        Get the force to apply based on the current step from the force schedule.
        """
        for start_step, end_step, force in self.force_schedule:
            if start_step <= self.count <= end_step:
                return np.array(force)
        return np.array([0.0, 0.0, 0.0])

    def modify_reward(self, reward):
        """
        Modify the reward before returning it.
        Example: Apply a scaling factor.
        """
        return reward
```
This is the wrapper for the best policy we had with a 10.3 N MSFOS. We compute a force schedule, given the time boundaries and the scales associated (see the reset method).

After the training, we get a policy in the form of .zip file then we launch `compute_msfos_policy.py`.

## Upkie Servos environment

Here again, we need to redefine a few things : 

```python
class VelocityEnvWrapper(gymnasium.Wrapper):
    """
    A custom wrapper for the velocity environment.

    This wrapper allows you to modify or log observations, rewards,
    or interactions with the environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.count = 0
        self.truncated = 0
        self.max_steps = 3000
        self.nb_servos = 6
        self.servos = list(self.action_space.keys())
        print(self.observation_space)
        print(self.action_space)
        self.obs_reg = np.array([ 1, 14,  2, 16, 10, 22,  1, 16,  2, 18, 13, 31.0])
        self.action_space = gymnasium.spaces.Box(low=-1.0,high=1.0,shape=(self.nb_servos,))
        self.observation_space = gymnasium.spaces.Box(low=-1.0,high=1.0,shape=(2*self.nb_servos,))

        

    def convert_obs(self,obs):
        new_observation = []
        for i in self.servos:
            new_observation.append(obs[i]["position"])
            new_observation.append(obs[i]["velocity"])
        obs = np.array(new_observation).flatten()
        obs = obs/self.obs_reg # all values should be in the [-1,1] range
        return obs


    def convert_act(self,act):
        action = {}
        for i in range(self.nb_servos) :
            if "wheel" in self.servos[i]:
                action_servo = {
                        "position": 0,
                        "velocity": act[i]*111,
                        "kp_scale": 0,
                        "kd_scale": 0.5,
                        }
            else:
                action_servo = {
                        "position": 0,
                        "velocity": act[i]*28,
                        "kp_scale": 0,
                        "kd_scale": 0.5,}
             
            action[self.servos[i]] = action_servo

        return action


    def reset(self, **kwargs):
        """
        Reset the environment and modify the initial observation if needed.
        """
        obs, info = self.env.reset(**kwargs)
        obs = self.convert_obs(obs)
        self.count = 0
        return obs, info

    

    def step(self, action):
        """
        Execute a step in the environment and modify the results if needed.
        """
        
        action = self.convert_act(action)
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.convert_obs(obs)
        body_height = info['spine_observation']['sim']['base']['position'][2]
        if body_height<0.35: # we have fallen
            print("fall ! ", self.count)
            reward = -10
            done = True
        else:
            reward = (body_height/0.70)**2
        self.count += 1
        return obs, reward, done, truncated, info
```

Let us explain some part of the code. First of all, the observations are now a bunch of parameters (position, velocity, torque etc...) for each servomotor. Each of these parameters have certain inf and sup boundaries. Therefore, we decided to normalize all of them using the `self.obs_reg` to assure that each observation is bounded between $[-1;1]$. That is the utility of the methods `convert_act` and `convert_obs`.

Another thing we changed is the reward given the height of the body. Indeed, when we did a try without this change of reward, and the body just touched the floor with its knees, so this change of reward keep it above the floor.

Lastly, we put the file `settings_servos.gin` where we specifically changed `PPOSettings.clip_range = 0.05` and the total number of steps `TrainingSettings.total_timesteps = 5_000_000`.

## Policies

In the folder policies/, you may find the three policies we trained for this project
* The base policy we trained with nothing added (no wrapper)
* The curriculum learning policy
* The servos policy

## Run servos policy

To run the servos policy, you will have to use `run_servos.py` file to have the good wrapper in order to launch the policy. Then launch it with :
```bash
python3 run_servos.py policy.zip
```