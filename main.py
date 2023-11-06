import csv
import time
import sys
from collections import deque
from memory_profiler import profile


class ReadINI:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        file_names = []
        repeats = []
        with open(self.file_path, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 3:
                    file_name = parts[0]
                    repeat = int(parts[1])
                    file_names.append(file_name)
                    repeats.append(repeat)
        return file_names, repeats


class ReadFile:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        num_ver = 0
        cost_list = []
        with open(self.file_path, "r") as file:
            num_ver = int(file.readline())
            for i in range(num_ver):
                distance = [int(x) for x in file.readline().split()]
                cost_list.append(distance)
        return cost_list


class TimeStamp:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        exec_time = self.end_time - self.start_time
        return exec_time


class SaveToFile:
    def __init__(self, file_name, search, repeats, cost, path, times):
        self.file_name = file_name
        self.search = search
        self.repeats = repeats
        self.cost = cost
        self.path = path
        self.times = times

    def save(self):
        with open('b&b_TSP_output.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            path_without_quotes = str(self.path).strip('"')
            path_without_spaces = path_without_quotes.replace(' ', '')
            writer.writerow([self.file_name, self.search, self.repeats, self.cost, path_without_spaces])
            writer.writerows([[str(time)] for time in self.times])


@profile(precision=4)
class Tsp_low_cost:
    def __init__(self, n, cost):
        self.n = n
        self.cost = cost
        self.min_dist = sys.maxsize
        self.best_path = None

    def calculate_total_distance(self, path):
        total_distance = sum(self.cost[path[i]][path[i + 1]] for i in range(self.n - 1))
        total_distance += self.cost[path[-1]][path[0]]
        return total_distance

    def solve_low_cost(self):
        if self.n <= 2:
            return list(range(self.n)), 0

        vertices = list(range(self.n))
        start_vertex = 0  # Start from vertex 0
        path = [start_vertex]
        remaining_vertices = vertices[:start_vertex] + vertices[start_vertex + 1:]
        lower_bound = self._calculate_lower_bound(path, remaining_vertices)

        self._tsp_branch_and_bound(path, remaining_vertices, 0, lower_bound)

        return int(self.min_dist), self.best_path

    def _calculate_lower_bound(self, path, remaining_vertices):
        lower_bound = 0
        # Add lower bound for the partial path
        for i in range(len(path) - 1):
            lower_bound += self.cost[path[i]][path[i + 1]]
        # Add lower bound for the nearest neighbor to the unvisited vertices
        if remaining_vertices:
            for vertex in remaining_vertices:
                costs = [int(self.cost[vertex][v]) for v in remaining_vertices if v != vertex]
                if costs:
                    min_cost = min(costs)
                else:
                    min_cost = 0
                lower_bound += min_cost
        return lower_bound

    def _tsp_branch_and_bound(self, path, remaining_vertices, current_distance, lower_bound):
        if not remaining_vertices:
            current_distance += self.cost[path[-1]][path[0]]
            if current_distance < self.min_dist:
                self.min_dist = current_distance
                self.best_path = path[:]
            return

        for next_vertex in remaining_vertices:
            if current_distance + self.cost[path[-1]][next_vertex] < self.min_dist:
                new_path = path + [next_vertex]
                new_remaining = [v for v in remaining_vertices if v != next_vertex]
                new_lower_bound = self._calculate_lower_bound(new_path, new_remaining)
                if new_lower_bound < self.min_dist:
                    self._tsp_branch_and_bound(new_path, new_remaining,
                                               current_distance + self.cost[path[-1]][next_vertex], new_lower_bound)

@profile(precision=4)
class TSP_breadth_search:
    def __init__(self, n, cost):
        self.n = n
        self.cost = cost
        self.min_dist = sys.maxsize
        self.best_path = None

    def calculate_total_distance(self, path):
        total_distance = sum(self.cost[path[i]][path[i + 1]] for i in range(self.n - 1))
        total_distance += self.cost[path[-1]][path[0]]
        return total_distance

    def solve_breadth_search(self):
        if self.n <= 2:
            return list(range(self.n)), 0

        vertices = list(range(self.n))
        start_vertex = 0  # Start from vertex 0
        path = [start_vertex]
        remaining_vertices = vertices[:start_vertex] + vertices[start_vertex + 1:]
        lower_bound = self._calculate_lower_bound(path, remaining_vertices)

        queue = deque()
        queue.append((path, remaining_vertices, 0, lower_bound))

        while queue:
            path, remaining_vertices, current_distance, lower_bound = queue.popleft()

            if current_distance > self.min_dist:
                continue

            if not remaining_vertices:
                current_distance += self.cost[path[-1]][path[0]]
                if current_distance < self.min_dist:
                    self.min_dist = current_distance
                    self.best_path = path[:]
                continue

            for next_vertex in remaining_vertices:
                if current_distance + self.cost[path[-1]][next_vertex] < self.min_dist:
                    new_path = path + [next_vertex]
                    new_remaining = [v for v in remaining_vertices if v != next_vertex]
                    new_lower_bound = self._calculate_lower_bound(new_path, new_remaining)
                    if new_lower_bound < self.min_dist:
                        queue.append((new_path, new_remaining, current_distance + self.cost[path[-1]][next_vertex],
                                      new_lower_bound))

        return int(self.min_dist), self.best_path

    def _calculate_lower_bound(self, path, remaining_vertices):
        lower_bound = 0
        # Add lower bound for the partial path
        for i in range(len(path) - 1):
            lower_bound += self.cost[path[i]][path[i + 1]]
        # Add lower bound for the nearest neighbor to the unvisited vertices
        if remaining_vertices:
            for vertex in remaining_vertices:
                costs = [int(self.cost[vertex][v]) for v in remaining_vertices if v != vertex]
                if costs:
                    min_cost = min(costs)
                else:
                    min_cost = 0
                lower_bound += min_cost
        return lower_bound

@profile(precision=4)
class TSP_depth_search:
    def __init__(self, n, cost):
        self.n = n
        self.cost = cost
        self.min_dist = sys.maxsize
        self.best_path = None

    def calculate_total_distance(self, path):
        total_distance = sum(self.cost[path[i]][path[i + 1]] for i in range(self.n - 1))
        total_distance += self.cost[path[-1]][path[0]]
        return total_distance

    def solve_depth_search(self):
        if self.n <= 2:
            return list(range(self.n)), 0

        vertices = list(range(self.n))
        start_vertex = 0  # Start from vertex 0
        path = [start_vertex]
        remaining_vertices = vertices[:start_vertex] + vertices[start_vertex + 1:]
        lower_bound = self._calculate_lower_bound(path, remaining_vertices)

        stack = [(path, remaining_vertices, 0, lower_bound)]

        while stack:
            path, remaining_vertices, current_distance, lower_bound = stack.pop()

            if current_distance > self.min_dist:
                continue

            if not remaining_vertices:
                current_distance += self.cost[path[-1]][path[0]]
                if current_distance < self.min_dist:
                    self.min_dist = current_distance
                    self.best_path = path[:]
                continue

            for next_vertex in remaining_vertices:
                if current_distance + self.cost[path[-1]][next_vertex] < self.min_dist:
                    new_path = path + [next_vertex]
                    new_remaining = [v for v in remaining_vertices if v != next_vertex]
                    new_lower_bound = self._calculate_lower_bound(new_path, new_remaining)
                    if new_lower_bound < self.min_dist:
                        stack.append((new_path, new_remaining, current_distance + self.cost[path[-1]][next_vertex],
                                      new_lower_bound))

        return int(self.min_dist), self.best_path

    def _calculate_lower_bound(self, path, remaining_vertices):
        lower_bound = 0
        # Add lower bound for the partial path
        for i in range(len(path) - 1):
            lower_bound += self.cost[path[i]][path[i + 1]]
        # Add lower bound for the nearest neighbor to the unvisited vertices
        if remaining_vertices:
            for vertex in remaining_vertices:
                costs = [int(self.cost[vertex][v]) for v in remaining_vertices if v != vertex]
                if costs:
                    min_cost = min(costs)
                else:
                    min_cost = 0
                lower_bound += min_cost
        return lower_bound


def main():
    times = []
    cost = 0
    path = []
    search = ["Low cost", "Breadth search", "Depth search"]

    reader = ReadINI("test.INI")
    file_paths, repeat = reader.read_data()

    for i in range(len(repeat)):
        file = ReadFile(file_paths[i])
        data = file.read_data()

        # Low cost
        algorithm = Tsp_low_cost(len(data), data)

        for j in range(repeat[i]):
            time_stamp = TimeStamp()
            time_stamp.start()
            cost, path = algorithm.solve_low_cost()
            exec_time = time_stamp.end()
            times.append(exec_time)
            print(search[0], " ", (i + 1), ".", (j + 1), " ", path, "cost: ", cost, " time: ", exec_time, "[s]")

        print()

        save_file = SaveToFile(file_paths[i], search[0], repeat[i], cost, path, times)
        save_file.save()

        times = []

        if i < 3:
            # Breadth search
            algorithm = TSP_breadth_search(len(data), data)

            for j in range(repeat[i]):
                time_stamp = TimeStamp()
                time_stamp.start()
                cost, path = algorithm.solve_breadth_search()
                exec_time = time_stamp.end()
                times.append(exec_time)
                print(search[1], " ", (i + 1), ".", (j + 1), " ", path, "cost: ", cost, " time: ", exec_time, "[s]")

            print()

            save_file = SaveToFile(file_paths[i], search[1], repeat[i], cost, path, times)
            save_file.save()

            times = []

            # Depth search
            algorithm = TSP_depth_search(len(data), data)

            for j in range(repeat[i]):
                time_stamp = TimeStamp()
                time_stamp.start()
                cost, path = algorithm.solve_depth_search()  # Lub solve_breadth_search() dla breadth-first search
                exec_time = time_stamp.end()
                times.append(exec_time)
                print(search[2], " ", (i + 1), ".", (j + 1), " ", path, "cost: ", cost, " time: ", exec_time, "[s]")

            print()

            save_file = SaveToFile(file_paths[i], search[2], repeat[i], cost, path, times)
            save_file.save()

            times = []


if __name__ == "__main__":
    main()
