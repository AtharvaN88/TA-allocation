import pulp
import pandas as pd
import numpy as np

# Load data
tas = pd.read_csv('tas.csv')
sections = pd.read_csv('sections.csv')

# Create the model
model = pulp.LpProblem("TA_Allocation", pulp.LpMinimize)

# Decision variables
x = pulp.LpVariable.dicts("assign",
                          ((i, j) for i in range(len(tas)) for j in range(len(sections))),
                          cat='Binary')

# Objective function
model += pulp.lpSum(x)  # Minimize total assignments (not really necessary, but helps reduce trivial solutions)

# Constraints
for i in range(len(tas)):
    # Max assignments per TA
    model += pulp.lpSum(x[i, j] for j in range(len(sections))) <= tas.loc[i, 'max_assigned']

for j in range(len(sections)):
    # Min TAs per section
    model += pulp.lpSum(x[i, j] for i in range(len(tas))) >= sections.loc[j, 'min_ta']
    # Max TAs per section
    model += pulp.lpSum(x[i, j] for i in range(len(tas))) <= sections.loc[j, 'max_ta']

# Preference constraints
for i in range(len(tas)):
    for j in range(len(sections)):
        if tas.iloc[i, j + 3] == 'U':
            model += x[i, j] == 0  # Unwilling assignments are not allowed

# Time conflict constraints
for i in range(len(tas)):
    for j1 in range(len(sections)):
        for j2 in range(j1 + 1, len(sections)):
            if sections.loc[j1, 'daytime'] == sections.loc[j2, 'daytime']:
                model += x[i, j1] + x[i, j2] <= 1

# Solve the model
model.solve()


# Function to calculate the score
def calculate_score(solution):
    score = 0
    for i in range(len(tas)):
        if sum(solution[i]) > tas.loc[i, 'max_assigned']:
            score += 1
    for j in range(len(sections)):
        if sum(solution[:, j]) < sections.loc[j, 'min_ta']:
            score += 1
    for i in range(len(tas)):
        for j in range(len(sections)):
            if solution[i, j] == 1:
                if tas.iloc[i, j + 3] == 'U':
                    score += 1
                elif tas.iloc[i, j + 3] == 'W':
                    score += 0.5
    return score


# Extract solutions
solutions = []
for i in range(len(tas)):
    row = []
    for j in range(len(sections)):
        row.append(int(x[i, j].value()))
    solutions.append(row)

solution_array = np.array(solutions)
score = calculate_score(solution_array)

print(f"Solution found with score: {score}")
print(solution_array)

# If you want to find multiple solutions, you can add constraints to exclude the current solution and re-solve
# This example finds up to 5 solutions with score <= 2
count = 1
while score <= 2 and count < 5:
    # Add constraint to exclude current solution
    model += pulp.lpSum(x[i, j] for i in range(len(tas)) for j in range(len(sections))
                        if solution_array[i, j] == 1) <= sum(sum(solution_array)) - 1

    # Solve again
    model.solve()

    # Extract new solution
    new_solution = []
    for i in range(len(tas)):
        row = []
        for j in range(len(sections)):
            row.append(int(x[i, j].value()))
        new_solution.append(row)

    new_solution_array = np.array(new_solution)
    new_score = calculate_score(new_solution_array)

    if new_score <= 2:
        count += 1
        print(f"\nAlternative solution {count} found with score: {new_score}")
        print(new_solution_array)
        solution_array = new_solution_array
        score = new_score
    else:
        break

print(f"\nFound {count} solutions with score <= 2")