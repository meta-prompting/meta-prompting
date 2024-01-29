# Description: A program to solve the game of 24
# Author: CR Agent XML v0.2

import pandas as pd
from itertools import permutations, product

# Function to parse puzzle string into a list of integers
def parse_puzzle(puzzle_str):
    return [int(num) for num in puzzle_str.split()]

# Operations: addition, subtraction, multiplication, division
operations = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y]
operation_symbols = ['+', '-', '*', '/']

# Function to apply operations to a set of numbers for a specific puzzle
def apply_operations_puzzle(numbers, operations):
    results = []
    for ops in product(operations, repeat=len(numbers)-1):
        for permutation in set(permutations(numbers)):
            # Building the expression with parentheses around each operation
            expression = f"({permutation[0]}{operation_symbols[operations.index(ops[0])]}{permutation[1]})"
            for i in range(1, len(numbers)-1):
                expression = f"({expression}{operation_symbols[operations.index(ops[i])]}{permutation[i+1]})"
            try:
                # Evaluating the expression
                if eval(expression) == 24:
                    results.append(expression)
            except ZeroDivisionError:
                continue
    return results

# Load the file
file_path = './24.csv'  # Replace with your file path
puzzle_df = pd.read_csv(file_path)

# Process and solve all puzzles
solutions = {}
for puzzle_str in puzzle_df['Puzzles']:
    puzzle_numbers = parse_puzzle(puzzle_str)
    solution = apply_operations_puzzle(puzzle_numbers, operations)
    solutions[puzzle_str] = solution

# Calculate the solved rate
solved_count = sum([1 for solution in solutions.values() if solution])
solved_rate = solved_count / len(solutions) * 100

print(f"Solved Rate: {solved_rate:.2f}%")

# If you want to save the solutions to a file
with open('solutions.txt', 'w') as file:
    for puzzle, solution in solutions.items():
        file.write(f"Puzzle: {puzzle}, Solutions: {solution}\n")
