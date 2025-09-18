"""
Title: test_allocation_TA.py
Purpose: Unit test for each objective functions used in the evolutionary computing framework
Group members: Kuan Chun Chiu, Atharva Nilapwar, Souren Prakash
Date: 2024/11/14
"""

import pytest
import numpy as np
from assignta import over_allocation, conflicts, under_support, unwilling, unpreferred, load_data

# Construct 3 functions wrapped with pytest.fixture decorator to initialize the 3 test cases
@pytest.fixture
def test1():
    sol1_path = "test1.csv"
    df = load_data(sol1_path)
    test1 = np.array(df)
    return test1

@pytest.fixture
def test2():
    sol2_path = "test2.csv"
    df = load_data(sol2_path)
    test2 = np.array(df)
    return test2

@pytest.fixture
def test3():
    sol3_path = "test3.csv"
    df = load_data(sol3_path)
    test3 = np.array(df)
    return test3

all_tests = [test1, test2, test3]
all_objectives_scores = {"over_allocation": [37, 41, 23], "conflicts": [8, 5, 2], "under_support": [1, 0, 7],
                         "unwilling": [53, 58, 43], "unpreferred": [15, 19, 10]}

# Test for the five objective functions using test 1, test 2, and test 3
for test in all_tests:
    idx = all_tests.index(test)
    def test_over_allocation(test):
        score = all_objectives_scores["over_allocation"][idx]
        ta_df = load_data("tas.csv")
        assert over_allocation(test, ta_df) == score, f"{score}, {over_allocation(test, ta_df)}"

    def test_conflicts(test):
        score = all_objectives_scores["conflicts"][idx]
        section_df = load_data("sections.csv")
        assert conflicts(test, section_df) == score, f"{score}, {conflicts(test, section_df)}"

    def test_under_support(test):
        score = all_objectives_scores["under_support"][idx]
        section_df = load_data("sections.csv")
        assert under_support(test, section_df) == score, f"{score}, {under_support(test, section_df)}"

    def test_unwilling(test):
        score = all_objectives_scores["unwilling"][idx]
        ta_df = load_data("tas.csv")
        assert unwilling(test, ta_df) == score, f"{score}, {unwilling(test, ta_df)}"

    def test_unpreferred(test):
        score = all_objectives_scores["unpreferred"][idx]
        ta_df = load_data("tas.csv")
        assert unpreferred(test, ta_df) == score, f"{score}, {unpreferred(test, ta_df)}"