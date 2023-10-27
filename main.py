import csv
import itertools
import time
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
    def __init__(self, file_name, repeats, cost, path, times):
        self.file_name = file_name
        self.repeats = repeats
        self.cost = cost
        self.path = path
        self.times = times

    def save(self):
        with open('b&b_TSP_output.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            path_without_quotes = str(self.path).strip('"')
            path_without_spaces = path_without_quotes.replace(' ', '')
            writer.writerow([self.file_name, self.repeats, self.cost, path_without_spaces])
            writer.writerows([[str(time)] for time in self.times])


class Tsp:
    def __init__(self, n, cost):
        self.n = n
        self.cost = cost
        self.min_dist = float('inf')
        self.best_path = None

    def calculate_total_distance(self, path):
        total_distance = sum(self.cost[path[i]][path[i + 1]] for i in range(self.n - 1))
        total_distance += self.cost[path[-1]][path[0]]
        return total_distance

    @profile(precision=4)
    def solve(self):
        if self.n <= 2:
            return list(range(self.n)), 0

        vertices = list(range(self.n))
        start_vertex = 0  # Start from vertex 0
        path = [start_vertex]
        remaining_vertices = vertices[:start_vertex] + vertices[start_vertex + 1:]
        self._tsp_branch_and_bound(path, remaining_vertices, 0)

        return int(self.min_dist), self.best_path

    def _tsp_branch_and_bound(self, path, remaining_vertices, current_distance):
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
                self._tsp_branch_and_bound(new_path, new_remaining, current_distance + self.cost[path[-1]][next_vertex])


def main():
    times = []
    cost = 0
    path = []

    reader = ReadINI("test.INI")
    file_paths, repeat = reader.read_data()

    for i in range(len(repeat)):
        file = ReadFile(file_paths[i])
        data = file.read_data()
        algorithm = Tsp(len(data), data)

        for j in range(repeat[i]):
            time_stamp = TimeStamp()
            time_stamp.start()
            cost, path = algorithm.solve()
            exec_time = time_stamp.end()
            times.append(exec_time)
            print(path, "cost: ", cost, " time: ", exec_time, "[s]")

        save_file = SaveToFile(file_paths[i], repeat[i], cost, path, times)
        save_file.save()

        times = []


if __name__ == "__main__":
    main()
