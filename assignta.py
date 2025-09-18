
"""
Title: allocate_TA.py
Purpose: Use the evolutionary computing framework to find the best solution to the TA allocation problem
Group members: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
Date: 2024/11/14
"""

import copy
from functools import reduce
from profiler import Profiler, profile
from datetime import datetime
import numpy as np
import pandas as pd
import random as rnd
from evo import Evo
import csv
import os

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

    """

Calculate the over-allocation score for a given solution.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA information with max assignments.

    Returns:
    int: The total over-allocation score, representing TAs assigned more sections than allowed.
    """
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

    """
    Calculate the number of conflicting TA assignments for sections that occur at the same time.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    section_df (pd.DataFrame): DataFrame containing section information with day/time data.

    Returns:
    int: The number of TAs with scheduling conflicts.
    """


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

    """
    Calculate the under-support score for sections that do not meet the minimum TA requirement.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    section_df (pd.DataFrame): DataFrame containing section information with minimum TA requirement.

    Returns:
    int: The total under-support score for all sections.
    """
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

    """
    Calculate the number of TA assignments where a TA was assigned to a section they marked as 'Unwilling'.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA preferences for each section.

    Returns:
    int: The total score for unwilling assignments.
    """
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

    """
    Calculate the number of TA assignments where a TA was assigned to a section they marked as 'Unpreferred'.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA preferences for each section.

    Returns:
    int: The total score for unpreferred assignments.
    """

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

    """
    Calculate the number of TA assignments where a TA was assigned to a section they marked as 'Unpreferred'.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA preferences for each section.

    Returns:
    int: The total score for unpreferred assignments.
    """
    new_solution = copy.deepcopy(solution)
    ta1, ta2 = rnd.sample(range(len(new_solution)), 2)
    section = rnd.randint(0, len(new_solution[0]) - 1)
    new_solution[ta1][section], new_solution[ta2][section] = new_solution[ta2][section], new_solution[ta1][section]
    return new_solution

@profile
def fix_overallocation_agent(solution, ta_df):
    """
    Modify the solution to reduce TA over-allocation by removing extra section assignments.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA information with max assignments.

    Returns:
    list: The modified solution with fewer over-allocated TAs.
    """
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

    """
    Resolve conflicts in the TA assignments by removing conflicting section assignments.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    section_df (pd.DataFrame): DataFrame containing section information with day/time data.

    Returns:
    list: The modified solution with reduced conflicts.

    """
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

    """
      Modify the solution to assign TAs to preferred sections instead of unpreferred ones.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA preferences for each section.

    Returns:
    list: The modified solution with more TAs assigned to preferred sections.
    """
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

    """
    Modify the solution to remove TA assignments to sections they are unwilling to teach.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    ta_df (pd.DataFrame): DataFrame containing TA preferences for each section.

    Returns:
    list: The modified solution with fewer TAs assigned to unwilling sections.
    """
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

    """

    Modify the solution to add more TAs to under-supported sections that do not meet the minimum TA requirement.

    Parameters:
    solution (list): A list of lists representing the TA allocation solution.
    section_df (pd.DataFrame): DataFrame containing section information with minimum TA requirement.

    Returns:
    list: The modified solution with more TAs assigned to under-supported sections.
    """
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




def create_summary_csv(population, runs_folder, timestamp):
    """
           Format and save the final population of solutions to a CSV file.

           Parameters:
           population (list): The final population of TA allocation solutions.

           Returns:
           None
        """
    filename = f"{runs_folder}/summary_{timestamp}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['groupname', 'overallocation', 'conflicts', 'undersupport', 'unwilling', 'unpreferred'])

        for eval, _ in population.items():
            row = ['wombats']
            for objective in eval:
                row.append(objective[1])
            writer.writerow(row)

    print(f"CSV file created: {filename}")


def save_best_solution(population, runs_folder, timestamp):
    filename = f"{runs_folder}/best_solution_{timestamp}.txt"

    best_eval = None
    best_sol = None
    lowest_score = float("inf")

    for eval, sol in population.items():
        final_score = sum([tuple[1] for tuple in eval])
        if final_score < lowest_score:
            lowest_score = final_score
            best_eval = eval
            best_sol = sol

    with open(filename, 'w') as f:
        f.write("Best solution found:\n\n")
        f.write(f"Evaluation: {best_eval}\n\n")
        f.write("Solution:\n")
        for row in best_sol:
            f.write(str(row) + "\n")

    print(f"Best solution details saved: {filename}")



def main():
    ta_df = load_data("tas.csv")
    section_df = load_data("sections.csv")
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
        print (f"{best_eval}: \n{best_sol}")

    initial_population = generate_initial_population(ta_df, section_df)
    for solution in initial_population:
        evo.add_solution(solution)
    time_limit = 300
    evo.evolve(time_limit=time_limit)
    Profiler.report()
    output_results(evo.population)


    runs_folder = "runs"
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_summary_csv(evo.population, runs_folder, timestamp)
    save_best_solution(evo.population, runs_folder, timestamp)


if __name__ == '__main__':

    main()