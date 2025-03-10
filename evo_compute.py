"""
Title: evo_compute.py
Purpose: Construct an evolutionary framework to find the best solution for the TA allocation problem.
This file specifically will be different than evo.py because it's retrofitted to use the computing power
of an 8 core gaming computer.
Group members: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
Date: 2024/11/14
"""

import random as rnd
import copy
from functools import reduce
from profiler import Profiler, profile
import numpy as np
import time
import multiprocessing

class Evo:
    def __init__(self):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Initialize the solution population, objectives, and agents state variables
        Parameter: N/A
        Return: N/A
        """
        self.population = {}
        self.objectives = {}
        self.agents = {}

    def add_objective(self, name, object_f):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Add an objective function to the objective dictionary with objective name as the key
        Parameter1: name (str), the name of the objective
        Parameter2: object_f (function), the objective function
        Return: N/A
        """
        self.objectives[name] = object_f

    def add_agents(self, name, agent_f, solution_num=1):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Add a tuple of agent function and solution numbers to the agents dictionary with agent name as the key
        Parameter1: name (str), the name of the agent
        Parameter2: agent_f (function), the agent function
        Parameter3: solution_num (int), the number of solutions used by the agent function to derive a new solution
        Return: N/A
        """
        self.agents[name] = (agent_f, solution_num)

    def add_solution(self, sol, evaluation=None):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Add a solution to the population dictionary with evaluation of the solution as the key, which is based
                 on each objective function in the objective dictionary.
        Parameter1: sol (array), a 2D array solution with rows being each TA and columns being the 17 lab sections
        Return: N/A
        """
        if evaluation is None:
            evaluation = tuple((name, object_f(sol)) for name, object_f in self.objectives.items())
        self.population[evaluation] = sol



    def get_random_solutions(self, solution_num=1):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Randomly select a number of solutions from the population dictionary depending on solution_num
        Parameter: solution_num (int), the number of solutions used by the agent function to derive a new solution
        Return: random_solutions (list), a list of randomly selected solutions
        """
        if len(self.population) == 0:
            random_solutions = []
        else:
            solutions = tuple(self.population.values())
            random_solutions = [copy.deepcopy(rnd.choice(solutions)) for _ in range(solution_num)]
        return random_solutions

    def run_agent(self, name):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Execute the specified agent function to derive a new solution and add it to the population
        Parameter: name (str), the name of the agent
        Return: N/A
        """
        agent_f, solution_num = self.agents[name]
        solutions = self.get_random_solutions(solution_num)
        new_solution = agent_f(solutions)
        self.add_solution(new_solution)

    def dominate(self, sol1, sol2):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Determine whether solution 1 dominates solution 2 or not, based on their objective evaluations
        Parameter1: sol1 (array), a 2D array solution with rows being each TA and columns being the 17 lab sections
        Parameter2: sol2 (array), a 2D array solution with rows being each TA and columns being the 17 lab sections
        Return: dom_boolean (boolean), True or False based on whether solution 1 dominates solution 2
        """
        condition_1 = (sol1[0] <= sol2[0] and sol1[1] <= sol2[1] and sol1[2] <= sol2[2]
                       and sol1[3] <= sol2[3] and sol1[4] < sol2[4])
        condition_2 = (sol1[0] <= sol2[0] and sol1[1] <= sol2[1] and sol1[2] <= sol2[2]
                       and sol1[3] < sol2[3] and sol1[4] <= sol2[4])
        condition_3 = (sol1[0] <= sol2[0] and sol1[1] <= sol2[1] and sol1[2] < sol2[2]
                       and sol1[3] <= sol2[3] and sol1[4] <= sol2[4])
        condition_4 = (sol1[0] <= sol2[0] and sol1[1] < sol2[1] and sol1[2] <= sol2[2]
                       and sol1[3] <= sol2[3] and sol1[4] <= sol2[4])
        condition_5 = (sol1[0] < sol2[0] and sol1[1] <= sol2[1] and sol1[2] <= sol2[2]
                       and sol1[3] <= sol2[3] and sol1[4] <= sol2[4])
        dom_boolean = condition_1 or condition_2 or condition_3 or condition_4 or condition_5
        return dom_boolean

    def reduce_non_dominated(self, S, sol):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Find all solutions from the population that are not dominated by sol
        Parameter1: S (set), a set of evaluations of all solutions in the population
        Parameter2: sol (array), a 2D array solution with rows being each TA and columns being the 17 lab sections
        Return: non_dominated (set), a set of evaluations of solutions that are not dominated by sol
        """
        dominated = {dom_sol for dom_sol in S if self.dominate(sol, dom_sol)}
        non_dominated = S - dominated
        return non_dominated

    def remove_dominated(self):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: Eliminate all dominated solutions in the population, so only non-dominated solutions are kept
        Parameter: N/A
        Return: N/A
        """
        non_dominated = reduce(self.reduce_non_dominated, self.population.keys(), self.population.keys())
        self.population = {evaluation: self.population[evaluation] for evaluation in non_dominated}

    @profile
    def evolve(self, time_limit=300, dom=100, status=1000):
        """
                Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
                Purpose: Keep running the agent functions to derive new solutions until the time limit is up.
                         Eliminate dominated solutions periodically, so only the best solutions are kept when the time is up
                Parameter1: time_limit (int), the maximum number of seconds allowed for the evolution to find the best solution
                Parameter2: dom (int), how often to eliminate dominated solutions
                Parameter3: status (int), how often to display the current population of solutions
                Return: N/A
        """
        start_time = time.time()
        agent_names = list(self.agents.keys())
        i = 0

        def run_agent_parallel(agent_name):
            agent_f, solution_num = self.agents[agent_name]
            solutions = self.get_random_solutions(solution_num)
            new_solution = agent_f(solutions)
            evaluation = tuple((name, object_f(new_solution)) for name, object_f in self.objectives.items())
            return new_solution, evaluation

        with multiprocessing.Pool(processes=6) as pool:
            while time.time() - start_time < time_limit:
                batch_size = 6  # Use 6 cores
                batch_agents = [rnd.choice(agent_names) for _ in range(batch_size)]
                results = pool.map(run_agent_parallel, batch_agents)

                for new_solution, evaluation in results:
                    self.population[evaluation] = new_solution

                i += batch_size

                if i % dom == 0:
                    self.remove_dominated()

                if i % status == 0:
                    self.remove_dominated()
                    elapsed_time = time.time() - start_time
                    print(f"Iteration: {i}")
                    print(f"Time elapsed: {elapsed_time:.2f} seconds")
                    print(f"Population size: {len(self.population)}")
                    print(self)

        self.remove_dominated()
        total_time = time.time() - start_time
        print(f"Evolution completed in {total_time:.2f} seconds\n")



    def __str__(self):
        """
        Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
        Purpose: To construct a string representation of the Evo object in terms of each solution in the population
        Parameter: N/A
        Return: result (str), a string representation of the solutions and their evaluations in the current population
        """
        result = ""
        for eval, sol in self.population.items():
            result += (f"{str(eval)}:   {str(sol)}\n")
        return result