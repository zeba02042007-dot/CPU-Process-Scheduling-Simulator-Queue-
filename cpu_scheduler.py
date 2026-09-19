"""
CPU Process Scheduling Simulator
================================
Implements five classic CPU scheduling algorithms:

  1. FCFS      - First Come First Served        (non-preemptive)
  2. SJF       - Shortest Job First             (non-preemptive)
  3. SRTF      - Shortest Remaining Time First  (preemptive SJF)
  4. Priority  - Priority Scheduling            (non-preemptive, lower number = higher priority)
  5. RR        - Round Robin                    (preemptive, time quantum)

For each algorithm the program prints:
  - a Gantt chart
  - a per-process table (Completion, Turnaround, Waiting, Response time)
  - average waiting / turnaround / response time, CPU utilization, throughput

Run:
    python cpu_scheduler.py              # run all algorithms on the sample data
    python cpu_scheduler.py --interactive  # enter your own processes
    python cpu_scheduler.py --quantum 3    # change the Round Robin time quantum
"""

import argparse
from collections import deque
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Data structure
# --------------------------------------------------------------------------
@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int = 0

    # filled in by the scheduler
    remaining: int = 0
    start: int = -1          # first time the process got the CPU
    completion: int = 0

    def reset(self):
        self.remaining = self.burst
        self.start = -1
        self.completion = 0

    @property
    def turnaround(self):    # TAT = Completion - Arrival
        return self.completion - self.arrival

    @property
    def waiting(self):       # WT = TAT - Burst
        return self.turnaround - self.burst

    @property
    def response(self):      # RT = First CPU time - Arrival
        return self.start - self.arrival


# --------------------------------------------------------------------------
# Helper: merge consecutive timeline slices into Gantt blocks
# --------------------------------------------------------------------------
def add_slice(gantt, pid, start, end):
    """Append a (pid, start, end) block, merging with the previous block if same pid."""
    if start == end:
        return
    if gantt and gantt[-1][0] == pid and gantt[-1][2] == start:
        gantt[-1] = (pid, gantt[-1][1], end)
    else:
        gantt.append((pid, start, end))


# --------------------------------------------------------------------------
# Scheduling algorithms  (each returns a Gantt list of (pid, start, end))
# --------------------------------------------------------------------------
def fcfs(processes):
    gantt, time = [], 0
    for p in sorted(processes, key=lambda p: (p.arrival, p.pid)):
        if time < p.arrival:
            add_slice(gantt, "IDLE", time, p.arrival)
            time = p.arrival
        p.start = time
        add_slice(gantt, p.pid, time, time + p.burst)
        time += p.burst
        p.completion = time
    return gantt


def sjf(processes):
    """Non-preemptive: among arrived processes, pick the shortest burst."""
    gantt, time, done = [], 0, 0
    n = len(processes)
    finished = set()
    while done < n:
        ready = [p for p in processes if p.arrival <= time and p.pid not in finished]
        if not ready:
            nxt = min(p.arrival for p in processes if p.pid not in finished)
            add_slice(gantt, "IDLE", time, nxt)
            time = nxt
            continue
        p = min(ready, key=lambda p: (p.burst, p.arrival, p.pid))
        p.start = time
        add_slice(gantt, p.pid, time, time + p.burst)
        time += p.burst
        p.completion = time
        finished.add(p.pid)
        done += 1
    return gantt


def srtf(processes):
    """Preemptive SJF: at every time unit run the process with least remaining time."""
    gantt, time, done = [], 0, 0
    n = len(processes)
    while done < n:
        ready = [p for p in processes if p.arrival <= time and p.remaining > 0]
        if not ready:
            add_slice(gantt, "IDLE", time, time + 1)
            time += 1
            continue
        p = min(ready, key=lambda p: (p.remaining, p.arrival, p.pid))
        if p.start == -1:
            p.start = time
        add_slice(gantt, p.pid, time, time + 1)
        p.remaining -= 1
        time += 1
        if p.remaining == 0:
            p.completion = time
            done += 1
    return gantt


def priority_scheduling(processes):
    """Non-preemptive: lowest priority number = highest priority."""
    gantt, time, done = [], 0, 0
    n = len(processes)
    finished = set()
    while done < n:
        ready = [p for p in processes if p.arrival <= time and p.pid not in finished]
        if not ready:
            nxt = min(p.arrival for p in processes if p.pid not in finished)
            add_slice(gantt, "IDLE", time, nxt)
            time = nxt
            continue
        p = min(ready, key=lambda p: (p.priority, p.arrival, p.pid))
        p.start = time
        add_slice(gantt, p.pid, time, time + p.burst)
        time += p.burst
        p.completion = time
        finished.add(p.pid)
        done += 1
    return gantt


def round_robin(processes, quantum=2):
    gantt, time = [], 0
    procs = sorted(processes, key=lambda p: (p.arrival, p.pid))
    queue, i, done, n = deque(), 0, 0, len(procs)

    def admit(now):
        nonlocal i
        while i < n and procs[i].arrival <= now:
            queue.append(procs[i])
            i += 1

    admit(time)
    while done < n:
        if not queue:                       # CPU idle until next arrival
            add_slice(gantt, "IDLE", time, procs[i].arrival)
            time = procs[i].arrival
            admit(time)
            continue
        p = queue.popleft()
        if p.start == -1:
            p.start = time
        run = min(quantum, p.remaining)
        add_slice(gantt, p.pid, time, time + run)
        time += run
        p.remaining -= run
        admit(time)                         # new arrivals join BEFORE the preempted process
        if p.remaining > 0:
            queue.append(p)
        else:
            p.completion = time
            done += 1
    return gantt


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def print_gantt(gantt):
    top = "|"
    bottom = ""
    for pid, s, e in gantt:
        width = max(len(pid) + 2, (e - s) * 2)
        top += pid.center(width) + "|"
        bottom += str(s).ljust(width + 1)
    bottom += str(gantt[-1][2])
    print("\nGantt Chart:")
    print(top)
    print(bottom)


def print_table(processes):
    print("\n{:<6}{:>8}{:>7}{:>9}{:>12}{:>12}{:>9}{:>10}".format(
        "PID", "Arrival", "Burst", "Priority", "Completion", "Turnaround", "Waiting", "Response"))
    print("-" * 73)
    for p in sorted(processes, key=lambda p: p.pid):
        print("{:<6}{:>8}{:>7}{:>9}{:>12}{:>12}{:>9}{:>10}".format(
            p.pid, p.arrival, p.burst, p.priority,
            p.completion, p.turnaround, p.waiting, p.response))


def compute_metrics(processes, gantt):
    n = len(processes)
    total_time = gantt[-1][2] - min(p.arrival for p in processes)
    busy = sum(e - s for pid, s, e in gantt if pid != "IDLE")
    return {
        "avg_wt": sum(p.waiting for p in processes) / n,
        "avg_tat": sum(p.turnaround for p in processes) / n,
        "avg_rt": sum(p.response for p in processes) / n,
        "cpu_util": busy / total_time * 100,
        "throughput": n / total_time,
    }


def run_algorithm(name, func, processes, **kwargs):
    for p in processes:
        p.reset()
    gantt = func(processes, **kwargs)
    m = compute_metrics(processes, gantt)

    print("\n" + "=" * 73)
    print(f"  {name}")
    print("=" * 73)
    print_gantt(gantt)
    print_table(processes)
    print(f"\n  Average Waiting Time    : {m['avg_wt']:.2f}")
    print(f"  Average Turnaround Time : {m['avg_tat']:.2f}")
    print(f"  Average Response Time   : {m['avg_rt']:.2f}")
    print(f"  CPU Utilization         : {m['cpu_util']:.1f}%")
    print(f"  Throughput              : {m['throughput']:.3f} processes/unit time")
    return m


def print_comparison(results):
    print("\n" + "=" * 73)
    print("  COMPARISON OF ALGORITHMS")
    print("=" * 73)
    print("{:<26}{:>13}{:>16}{:>14}".format("Algorithm", "Avg Waiting", "Avg Turnaround", "Avg Response"))
    print("-" * 69)
    for name, m in results.items():
        print("{:<26}{:>13.2f}{:>16.2f}{:>14.2f}".format(
            name, m["avg_wt"], m["avg_tat"], m["avg_rt"]))
    best = min(results, key=lambda k: results[k]["avg_wt"])
    print(f"\n  Lowest average waiting time: {best}")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def sample_processes():
    return [
        Process("P1", arrival=0, burst=8, priority=3),
        Process("P2", arrival=1, burst=4, priority=1),
        Process("P3", arrival=2, burst=9, priority=4),
        Process("P4", arrival=3, burst=5, priority=2),
    ]


def read_processes():
    n = int(input("Number of processes: "))
    procs = []
    for i in range(1, n + 1):
        print(f"--- Process P{i} ---")
        a = int(input("  Arrival time : "))
        b = int(input("  Burst time   : "))
        pr = int(input("  Priority (lower = higher priority): "))
        procs.append(Process(f"P{i}", a, b, pr))
    return procs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CPU Process Scheduling Simulator")
    parser.add_argument("--interactive", action="store_true", help="enter your own processes")
    parser.add_argument("--quantum", type=int, default=2, help="Round Robin time quantum (default 2)")
    args = parser.parse_args()

    processes = read_processes() if args.interactive else sample_processes()

    print("CPU PROCESS SCHEDULING SIMULATOR")
    print("Input processes:", ", ".join(
        f"{p.pid}(AT={p.arrival}, BT={p.burst}, PR={p.priority})" for p in processes))

    results = {}
    results["FCFS"] = run_algorithm("FCFS - First Come First Served", fcfs, processes)
    results["SJF"] = run_algorithm("SJF - Shortest Job First (non-preemptive)", sjf, processes)
    results["SRTF"] = run_algorithm("SRTF - Shortest Remaining Time First", srtf, processes)
    results["Priority"] = run_algorithm("Priority Scheduling (non-preemptive)", priority_scheduling, processes)
    results[f"Round Robin (q={args.quantum})"] = run_algorithm(
        f"Round Robin (quantum = {args.quantum})", round_robin, processes, quantum=args.quantum)

    print_comparison(results)


if __name__ == "__main__":
    main()
