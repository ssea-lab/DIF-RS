# coding=gbk
import numpy as np

from algorithms.objective import get_objective


class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_reward = np.inf


class ParticleSwarmOptimization:
    def __init__(self, task_list):
        self.task_list = task_list
        self.particle_n = 50
        self.iteration_n = 300
        self.inertia = 0.5
        self.cognitive_weight = 1.0
        self.social_weight = 1.0
        self.task_n = task_list.shape[0]
        self.particles = []
        self.global_best_position = None
        self.global_best_reward = np.inf

    def run(self):
        self.initialize_particles()
        self.global_best_position = self.particles[0].position
        self.global_best_reward = self.calculate_path_reward(self.global_best_position)
        for _ in range(self.iteration_n):
            for particle in self.particles:
                particle.velocity = self.update_velocity(particle)
                particle.position = self.update_position(particle)
                particle_reward = self.calculate_path_reward(particle.position)
                if particle_reward < particle.best_reward:
                    particle.best_reward = particle_reward
                    particle.best_position = particle.position
                if particle_reward < self.global_best_reward:
                    self.global_best_reward = particle_reward
                    self.global_best_position = particle.position
        return self.global_best_position, self.global_best_reward

    def initialize_particles(self):
        for _ in range(self.particle_n):
            position = np.random.permutation(self.task_n)
            particle = Particle(position)
            self.particles.append(particle)

    def update_velocity(self, particle):
        cognitive_component = self.cognitive_weight * np.random.rand() * (particle.best_position - particle.position)
        social_component = self.social_weight * np.random.rand() * (self.global_best_position - particle.position)
        velocity = self.inertia * particle.velocity + cognitive_component + social_component
        return velocity

    def update_position(self, particle):
        position = np.argsort(np.argsort(particle.position + particle.velocity))
        return position

    def calculate_path_reward(self, path):
        objective = get_objective(self.task_list, path)
        return objective.reward


def get_idx_list(task_list, dataset):
    """
    粒子群算法
    初始粒子数：50
    迭代轮数：300
    """
    pso = ParticleSwarmOptimization(task_list)
    best_path, _ = pso.run()
    return best_path
