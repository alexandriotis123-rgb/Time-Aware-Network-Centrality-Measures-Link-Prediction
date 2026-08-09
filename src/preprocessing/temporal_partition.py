

"""
PART 1 QUESTION 1
Temporal partition of the Stack Overflow network.
"""

import pandas as pd
import networkx as nx


def partition_temporal_network(
    df: pd.DataFrame,
    num_periods: int):
    """
    Partition the complete temporal network into N non-overlapping
    time periods.

    """

    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()

    delta_T = t_max - t_min

    delta_t = delta_T / num_periods

    time_points = []

    for j in range(num_periods + 1):
        t_j = t_min + j * delta_t
        time_points.append(t_j)

    time_periods = []

    for j in range(num_periods):
        start = time_points[j]
        end = time_points[j + 1]

        time_periods.append((start, end))

    print(f"t_min = {t_min}")
    print(f"t_max = {t_max}")
    print(f"ΔT = {delta_T}")
    print(f"Number of periods (N) = {num_periods}")
    print(f"δt = {delta_t}")

    print("\nTime points:")
    for t in time_points:
        print(t)

    print("\nTime periods:")
    for i, (start, end) in enumerate(time_periods, start=1):
        print(f"T{i}: [{start}, {end}]")

    subgraphs = []

    for j, (start, end) in enumerate(time_periods):

        if j < num_periods - 1:

            period_df = df[
                (df["timestamp"] >= start) &
                (df["timestamp"] < end)]

        else:

            period_df = df[
                (df["timestamp"] >= start) &
                (df["timestamp"] <= end)]

        graph = nx.from_pandas_edgelist(
            period_df,
            source="source",
            target="target",
            create_using=nx.Graph())

        graph.remove_edges_from(nx.selfloop_edges(graph))

        subgraphs.append(graph)

    return time_points, time_periods, subgraphs
