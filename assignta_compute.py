"""
Title: allocate_TA.py
Purpose: Use the evolutionary computing framework to find the best solution to the TA allocation problem
Group members: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
Date: 2024/11/14
"""

import copy
from functools import reduce
from profiler import Profiler, profile
import numpy as np
import pandas as pd
import random as rnd
from evo import Evo
import csv
import os
import time
from datetime import datetime

# Extract the ta availability info
def load_data(file_path):
    """
    Contributors: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
    Purpose: Convert the csv file into a dataframe
    Parameter: file_path (str), the file path of the csv file
    Return: df (data frame), the data frame containing the csv file data
    """
    test_cases = ["test1.csv", "test2.csv", "test3.csv"]
    if file_path in test_cases:
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path)
    return df


# Construct the five objective functions
def over_allocation(solution, ta_df):
    ta_assignment_max = list(ta_df["max_assigned"])
    total_assigned = []
    for ta in solution:
        assigned = list(ta).count(1)
        total_assigned.append(assigned)
    max_and_assigned = zip(ta_assignment_max, total_assigned)
    allocation_score_list = [assigned - max for max, assigned in max_and_assigned if assigned > max]
    allocation_score = sum(allocation_score_list)
    return allocation_score


def conflicts(solution, section_df):
    section_daytimes = list(section_df["daytime"])
    conflict_count = 0
    for ta_assignments in solution:
        assigned_sections = [i for i, assigned in enumerate(ta_assignments) if assigned == 1]
        assigned_daytimes = [section_daytimes[i] for i in assigned_sections]
        day_groups = {}
        for daytime in assigned_daytimes:
            day, time = daytime.split(' ')
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(time)
        has_conflict = False
        for day, times in day_groups.items():
            if len(times) != len(set(times)):
                has_conflict = True
                break
        if has_conflict:
            conflict_count += 1
    return conflict_count


def under_support(solution, section_df):
    min_ta = list(section_df["min_ta"])
    solution_df = pd.DataFrame(solution)
    all_total_ta = []
    for section in solution_df.columns:
        total_ta = sum(solution_df[section])
        all_total_ta.append(total_ta)
    min_and_assigned = zip(min_ta, all_total_ta)
    under_support_list = [abs(assigned - min) for min, assigned in min_and_assigned if assigned < min]
    under_support_score = sum(under_support_list)
    return under_support_score


def unwilling(solution, ta_df):
    all_assigned_sections = []
    for ta in solution:
        ta = list(ta)
        assigned_sections = []
        idx = 0
        for availability in ta:
            if availability == 1:
                assigned_sections.append(idx)
            idx += 1
        all_assigned_sections.append(assigned_sections)
    unwilling_scores = 0
    for row_idx in range(len(ta_df)):
        preference = ta_df.iloc[row_idx, 3:]
        ta_sections = all_assigned_sections[row_idx]
        if len(ta_sections) == 0:
            continue
        else:
            preference = [preference[idx] for idx in ta_sections]
            unwilling_score = preference.count("U")
            unwilling_scores += unwilling_score
    return unwilling_scores


def unpreferred(solution, ta_df):
    all_assigned_sections = []
    for ta in solution:
        ta = list(ta)
        assigned_sections = []
        idx = 0
        for availability in ta:
            if availability == 1:
                assigned_sections.append(idx)
            idx += 1
        all_assigned_sections.append(assigned_sections)
    unpreferred_scores = 0
    for row_idx in range(len(ta_df)):
        preference = ta_df.iloc[row_idx, 3:]
        ta_sections = all_assigned_sections[row_idx]
        if len(ta_sections) == 0:
            continue
        else:
            preference = [preference[idx] for idx in ta_sections]
            unpreferred_score = preference.count("W")
            unpreferred_scores += unpreferred_score
    return unpreferred_scores


# Construct the four agent functions
@profile
def random_swap_agent(solution):
    new_solution = copy.deepcopy(solution)
    ta1, ta2 = rnd.sample(range(len(new_solution)), 2)
    section = rnd.randint(0, len(new_solution[0]) - 1)
    new_solution[ta1][section], new_solution[ta2][section] = new_solution[ta2][section], new_solution[ta1][section]
    return new_solution


@profile
def fix_overallocation_agent(solution, ta_df):
    new_solution = copy.deepcopy(solution)
    max_assigned = ta_df["max_assigned"].tolist()
    for ta_idx, ta_assignments in enumerate(new_solution):
        current_assignments = sum(ta_assignments)
        if current_assignments > max_assigned[ta_idx]:
            original_sections = [idx for idx, section in enumerate(ta_assignments) if section == 1]
            sections_to_remove = rnd.sample(original_sections, current_assignments - max_assigned[ta_idx])
            for section in sections_to_remove:
                new_solution[ta_idx][section] = 0
    return new_solution


@profile
def fix_conflicts_agent(solution, section_df):
    new_solution = copy.deepcopy(solution)
    section_times = section_df["daytime"].tolist()
    for ta_idx, ta_assignments in enumerate(new_solution):
        assigned_sections = [i for i, assigned in enumerate(ta_assignments) if assigned == 1]
        assigned_times = [section_times[i] for i in assigned_sections]
        if len(assigned_times) != len(set(assigned_times)):
            conflicting_section = rnd.choice(assigned_sections)
            new_solution[ta_idx][conflicting_section] = 0
    return new_solution


@profile
def fix_preferences_agent(solution, ta_df):
    new_solution = copy.deepcopy(solution)
    for ta_idx, ta_assignments in enumerate(new_solution):
        preferences = ta_df.iloc[ta_idx, 3:].tolist()
        assigned_sections = [i for i, assigned in enumerate(ta_assignments) if assigned == 1]
        unpreferred_sections = [i for i in assigned_sections if preferences[i] == "W"]
        if unpreferred_sections:
            section_to_change = rnd.choice(unpreferred_sections)
            preferred_sections = [i for i, pref in enumerate(preferences)
                                  if pref == "P" and new_solution[ta_idx][i] == 0]
            if preferred_sections:
                new_section = rnd.choice(preferred_sections)
                new_solution[ta_idx][section_to_change] = 0
                new_solution[ta_idx][new_section] = 1
    return new_solution


@profile
def fix_unwilling_agent(solution, ta_df):
    new_solution = copy.deepcopy(solution)
    for ta_idx, ta_assignments in enumerate(new_solution):
        preferences = ta_df.iloc[ta_idx, 3:].tolist()
        assigned_sections = [i for i, assigned in enumerate(ta_assignments) if assigned == 1]
        unwilling_sections = [i for i in assigned_sections if preferences[i] == "U"]
        if unwilling_sections:
            section_to_change = rnd.choice(unwilling_sections)
            willing_sections = [i for i, pref in enumerate(preferences)
                                if pref in ["W", "P"] and new_solution[ta_idx][i] == 0]
            if willing_sections:
                new_section = rnd.choice(willing_sections)
                new_solution[ta_idx][section_to_change] = 0
                new_solution[ta_idx][new_section] = 1
    return new_solution


@profile
def fix_under_support_agent(solution, section_df):
    new_solution = copy.deepcopy(solution)
    min_ta = section_df["min_ta"].tolist()

    current_assignments = [sum(ta[section] for ta in new_solution) for section in range(len(new_solution[0]))]
    under_supported = [i for i, (current, minimum) in enumerate(zip(current_assignments, min_ta)) if current < minimum]

    if under_supported:
        section_to_fix = rnd.choice(under_supported)
        available_tas = [i for i, ta in enumerate(new_solution) if ta[section_to_fix] == 0]

        while current_assignments[section_to_fix] < min_ta[section_to_fix] and available_tas:
            ta_to_assign = rnd.choice(available_tas)
            new_solution[ta_to_assign][section_to_fix] = 1
            current_assignments[section_to_fix] += 1
            available_tas.remove(ta_to_assign)

    return new_solution


def main(num_runs=1):
    ta_df = load_data("tas.csv")
    section_df = load_data("sections.csv")

    if not os.path.exists("tests"):
        os.makedirs("tests")

    for run in range(num_runs):
        print(f"\nRun {run + 1} of {num_runs}")

        evo = Evo()

        evo.add_objective("over_allocation", lambda sol: over_allocation(sol, ta_df))
        evo.add_objective("conflicts", lambda sol: conflicts(sol, section_df))
        evo.add_objective("under_support", lambda sol: under_support(sol, section_df))
        evo.add_objective("unwilling", lambda sol: unwilling(sol, ta_df))
        evo.add_objective("unpreferred", lambda sol: unpreferred(sol, ta_df))

        evo.add_agents("random_swap_agent", lambda sols: random_swap_agent(sols[0]))
        evo.add_agents("fix_overallocation", lambda sols: fix_overallocation_agent(sols[0], ta_df))
        evo.add_agents("fix_conflicts", lambda sols: fix_conflicts_agent(sols[0], section_df))
        evo.add_agents("fix_preferences", lambda sols: fix_preferences_agent(sols[0], ta_df))
        evo.add_agents("fix_unwilling", lambda sols: fix_unwilling_agent(sols[0], ta_df))
        evo.add_agents("fix_under_support", lambda sols: fix_under_support_agent(sols[0], section_df))

        def generate_initial_population(ta_df, section_df):
            num_tas = len(ta_df)
            num_sections = len(section_df)
            population_size = 50
            population = []
            for _ in range(population_size):
                solution = [[rnd.choice([0, 1]) for _ in range(num_sections)] for _ in range(num_tas)]
                population.append(solution)
            return population

        def output_results(population):
            print("\nBest solutions found:")
            lowest_score = float("inf")
            for eval, sol in population.items():
                final_score = sum([tuple[1] for tuple in eval])
                if final_score < lowest_score:
                    lowest_score = final_score
                    best_eval = eval
                    best_sol = sol
            print(f"{best_eval}: \n{best_sol}")


        initial_population = generate_initial_population(ta_df, section_df)
        for solution in initial_population:
            evo.add_solution(solution)

        time_limit = 300
        evo.evolve(time_limit=time_limit)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tests/run_{run + 1}_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(f"Run {run + 1} of {num_runs}\n\n")
            f.write("Best solutions found:\n")
            for eval, sol in evo.population.items():
                f.write(f"{eval}: \n{sol}\n\n")

        print(f"Results saved to {filename}")

    Profiler.report()


if __name__ == '__main__':
    import sys

    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(num_runs=4)