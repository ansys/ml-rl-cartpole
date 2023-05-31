import collections
import logging
import os

import gym
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt


def basic_diagnostics(episode, steps, r_tot, r_max, r_ave, r_ave_max, steps_tot):
    print(
        "e: %5d n: %4d x: %s r: %s v: %s vx: %s nt: %d"
        % (
            episode,
            steps,
            "{:6.1f}".format(r_max),
            "{:6.1f}".format(r_tot),
            "{:6.1f}".format(r_ave),
            "{:6.1f}".format(r_ave_max),
            steps_tot,
        )
    )


def basic_epsilon_constant(epsilon):
    while True:
        yield epsilon


def basic_epsilon_linear(epsilon_start, epsilon_end, n_steps):
    slope = (epsilon_end - epsilon_start) / n_steps
    for i in range(n_steps):
        yield epsilon_start + i * slope

    while True:
        yield epsilon_end


def run(
    env_name,
    learner_factory,
    regression_factory,
    layers,
    n_episodes,
    epsilon,
    gamma,
    n_mini_batch,
    replay_db_warmup,
    replay_db_capacity,
    c_cycle,
    polyak_rate,
    averaging_window,
    victory_threshold,
    double_dqn=True,
    potential_fn=None,
    diagnostics_fn=None,
    output_path=None,
    output_name=None,
):
    env = gym.make(env_name)
    env_db = gym.make(env_name)

    learner = learner_factory(
        env_db,
        regression_factory,
        layers=layers,
        epsilon=epsilon,
        gamma=gamma,
        n_mini_batch=n_mini_batch,
        replay_db_warmup=replay_db_warmup,
        replay_db_capacity=replay_db_capacity,
        c_cycle=c_cycle,
        polyak_rate=polyak_rate,
        shaper_fn=potential_fn,
        output_path=output_path,
        output_name=output_name,
        double_dqn=double_dqn,
        diagnostics=False,
    )

    def action_sampler():
        return env.action_space.sample()

    episode_final = 0
    r_max = r_ave_max = float("-inf")
    r_tot = 0
    r_ave = 0
    steps_tot = 0
    rewards = np.zeros(n_episodes, dtype=float)
    master_results = np.zeros((n_episodes, 7), dtype=float)

    for episode in range(n_episodes):
        s0 = env.reset()
        learner.start_state(s0)

        done, r_tot, steps = False, 0.0, 0
        while not done:
            a = learner.next_action(action_sampler)
            sp, r, done, info = env.step(a)
            learner.next_reading(sp, r, done)
            r_tot += r
            steps += 1

            if "pyansys-CartPole-v0" in env_name:  # pyansys
                theta = env.env.env._theta_deg
                velo = env.env.env._cart_velocity
            else:
                theta = env.env.state[2]
                velo = env.env.state[1]

            logging.info(
                f"Action: {a:2.0f}\tReward: {r:3.0f}\tStep: {steps:3.0f}\t"
                f"Total reward: {r_tot:3.0f}\tTheta: {theta:12.6f}\tVelocity: {velo:12.6f}"
            )

        steps_tot += steps
        rewards[episode] = r_tot
        r_max = max(r_max, r_tot)
        r_ave = np.mean(rewards[max(0, episode + 1 - averaging_window) : episode + 1])
        r_ave_max = max(r_ave_max, r_ave)
        master_results[episode] = [
            episode,
            steps,
            r_tot,
            r_max,
            r_ave,
            r_ave_max,
            steps_tot,
        ]
        if diagnostics_fn is not None:
            diagnostics_fn(episode, steps, r_tot, r_max, r_ave, r_ave_max, steps_tot)
        if r_ave >= victory_threshold:
            episode_final = episode
            break

    if output_name:
        learner.save(output_path, output_name)
        np.save(os.path.join(output_path, output_name + "_master.npy"), master_results)
    return episode_final, r_max, r_tot, r_ave, r_ave_max, steps_tot


from IPython.display import clear_output


class LivePlotter:
    def __init__(self):
        self.data = collections.defaultdict(list)
        self.figsize = (18, 8)
        self.title = "Training History"
        self.figure, self.ax = plt.subplots(1, figsize=self.figsize)

    def live_plot(self, episode, steps, r_tot, r_max, r_ave, r_ave_max, steps_tot):
        self.data["current"].append(r_tot)
        self.data["average"].append(r_ave)

        clear_output(wait=True)
        plt.plot(self.data["current"], label="current", color="orange", linestyle="dashed")
        plt.plot(self.data["average"], label="average", color="blue", linestyle="-")

        if r_ave > 196:
            plt.figure(1).text(
                0.95,
                0.05,
                "Success!",
                fontsize=50,
                color="gray",
                ha="right",
                va="bottom",
                alpha=0.5,
            )

        plt.title(self.title)
        plt.grid(True)
        plt.xlabel("episode")
        plt.ylabel("reward")
        if len(self.data["current"]) == 1:
            plt.legend(loc="lower right")  # the plot evolves to the right
        plt.ylim([0, 200])
        plt.xlim([0, 300])
        plt.show()
