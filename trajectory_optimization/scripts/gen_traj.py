import numpy as np
import wandb

from trajectory_optimization.utils import load_model_and_instantiate_env

if __name__ == "__main__":
    NB_STEPS = 100_000
    TRAJ_LEN = 50
    NB_EPISODES = NB_STEPS // TRAJ_LEN

    run = wandb.init(project="skyline", job_type="gen_traj")
    model, env = load_model_and_instantiate_env(
        artifact_alias="agent:v38",
        time_limit=TRAJ_LEN,
        max_wheels_out=4,  # render_mode="human"
    )

    # trajectory at each step should contain
    # pos_x, pos_y, yaw, speed, steering_command in deg, speed_command in m/s
    trajectories = []

    obs, truncated, terminated = None, False, False
    current_traj = []
    while len(trajectories) < NB_EPISODES:
        if obs is None or (truncated or terminated):
            obs, _ = env.reset()
            if truncated:  # means the car successfully ran TRAJ_LEN steps without crashing
                current_traj[-1][-1] = True  # set the last traj_step to be done.
                # the done flag is offset by one, meaning its actually the next step that's done
                # it shouldn't be a problem for what we do + this is so the action isn't offset
                trajectories.append(current_traj)
                print(f"{len(trajectories)} / {NB_EPISODES}")
            current_traj = []

        action, _states = model.predict([obs], deterministic=True)
        action = action[0]  # action[0] bc model was trained on vec env

        car = env.unwrapped.car
        # rescaled_action = info["rescaled_action"]
        # done = terminated or truncated
        traj_step = [
            car.pos_x,
            car.pos_y,
            car.yaw,
            car.speed,
            # rescaled_action[
            #     0
            # ],
            action[
                0
            ],  # let's just grab the scaled action, could later rescale it (if we logged the output traj to wandb)
            # by fetching config that produced the agent
            # ultimately rn we want to train a nn to predict this value so -1, 1 will do
            # this script is a lil messy and a bit specific but eh
            action[1],
            False,
        ]
        current_traj.append(traj_step)
        obs, reward, terminated, truncated, info = env.step(action)

    trajectories = np.array(trajectories)
    # Reshape the 3D trajectories (n_traj, n_steps, traj_step_len) array to a 2D array (n_traj*n_step, traj_step_len)
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    np.savetxt("../data/longi_rl_trajectories.txt", trajectories)

    # TODO save to wandb artifacts
    run.finish()
