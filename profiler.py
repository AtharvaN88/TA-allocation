"""
profiler.py
A profiler class that demonstrates the use of decorators to support code profiling
DS3500: Advanced Programming with Data (Prof. Rachlin)
"""

from collections import defaultdict
import time

def profile(f):
    """ Convenience function to make decorator tags simpler:
        e.g. @profile instead of @Profiler.profile """
    return Profiler.profile(f)

class Profiler:
    """ A code profiling class.  Keeps track of function calls and running time. """

    calls = defaultdict(int)  # default = 0
    time = defaultdict(float)  # default = 0.0

    @staticmethod
    def _add(function_name, sec):
        """ Add 1 call and <sec> time to named function tracking """
        Profiler.calls[function_name] += 1
        Profiler.time[function_name] += sec

    @staticmethod
    def profile(f):
        """ The profiling decorator """
        def wrapper(*args, **kwargs):
            function_name = str(f).split()[1]
            start = time.time_ns()
            val = f(*args, **kwargs)
            sec = (time.time_ns() - start) / 10**9
            Profiler._add(function_name, sec)
            return val
        return wrapper

    @staticmethod
    def report():
        """ Summarize # calls, total runtime, and time/call for each function """
        print(f'{"Function":<25} {"Calls":<8} {"TotSec":<12} {"Sec/Call":<10}')
        print("-" * 56)  # Separator line for readability
        for name, num in Profiler.calls.items():
            sec = Profiler.time[name]
            print(f'{name:<25} {num:<8d} {sec:<12.6f} {sec / num:<10.6f}')
