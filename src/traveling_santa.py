import logging
from math import sqrt
from os.path import isfile
from time import time

from dataclasses import dataclass
from scipy.sparse import load_npz, lil_matrix, save_npz

N_CPU = 4
logging.basicConfig(level=logging.INFO)


def primes_sieve(limit):
    a = [True] * limit
    a[0] = a[1] = False
    out = []
    for (i, is_prime) in enumerate(a):
        if is_prime:
            for n in range(i * i, limit, i):
                a[n] = False
            out.append(i)
    return out


@dataclass
class City:
    id: int
    x: float
    y: float
    is_prime: bool

    @staticmethod
    def dist(a: "City", b: "City"):
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class Region:
    def __init__(self, bottom_left, top_right, cities):
        self.bottom_left = bottom_left
        self.top_right = top_right
        for city in cities:
            assert all([city.x >= self.bottom_left[0],
                        city.y >= self.bottom_left[1],
                        city.x <= self.top_right[0],
                        city.y <= self.top_right[1]]), f"city outside of boundaries"
        self.cities = cities

    def split(self):
        def get_dimension(city):
            if vertical:
                return city.x
            return city.y

        vertical = False
        if (self.top_right[0] - self.bottom_left[0]) >= (self.top_right[1] - self.bottom_left[1]):
            vertical = True
        self.cities.sort(key=get_dimension)
        split_id = int(len(self.cities) / 2)
        split_location = 0.5 * (self.cities[split_id - 1].x + self.cities[split_id].x if vertical else
                                self.cities[split_id - 1].y + self.cities[split_id].y)
        return (Region(self.bottom_left, (split_location, self.top_right[1]), self.cities[:split_id]),
                Region((split_location, self.bottom_left[1]), self.top_right, self.cities[split_id:])) \
            if vertical else \
            (Region(self.bottom_left, (self.top_right[0], split_location), self.cities[:split_id]),
             Region((self.bottom_left[0], split_location), self.top_right, self.cities[split_id:]))

    @staticmethod
    def is_neighbour(a: "Region", b: "Region"):
        return any([all([b.top_right[1] == a.bottom_left[1],
                         b.top_right[0] >= a.bottom_left[0],
                         b.bottom_left[0] <= a.top_right[0]]),
                    all([b.bottom_left[0] == a.top_right[0],
                         b.top_right[1] >= a.bottom_left[1],
                         b.bottom_left[1] <= a.top_right[1]]),
                    all([b.bottom_left[1] == a.top_right[1],
                         b.top_right[0] >= a.bottom_left[0],
                         b.bottom_left[0] <= a.top_right[0]]),
                    all([b.top_right[0] == a.bottom_left[0],
                         b.top_right[1] >= a.bottom_left[1],
                         b.bottom_left[1] <= a.top_right[1]])
                    ])


class CityMap:
    def __init__(self, data_path='../data/cities.csv', max_cities: int = 10):
        with open(data_path, 'r') as file:
            lines = file.readlines()
        lines.pop(0)
        self.primes = primes_sieve(len(lines))
        self.cities = []
        for id_, line in enumerate(lines):
            line = line.split(',')
            self.cities.append(City(int(line[0]), float(line[1]), float(line[2]), id_ in self.primes))
        self.n_cities = len(self.cities)
        self.distance_matrix = None
        self.regions = []
        if isfile('../store/dist_matrix.matrix.npz'):
            self.distance_matrix = load_npz('../store/dist_matrix.matrix.npz')
        else:
            self.build_regions(max_cities)
            self.build_distance_matrix()

    def map_as_data(self):
        x = [0] * self.n_cities
        y = [0] * self.n_cities
        for i, city in enumerate(self.cities):
            x[i] = city.x
            y[i] = city.y
        return x, y

    def score_route(self, route=None):
        logging.info(f"scoring route")
        if route is None:
            route = range(self.n_cities)
        if route[0]:
            start = route.index(min(route))
            route = route[start:] + route[:start]
        score = 0
        since_last_prime = 0
        for idx in range(len(route)):
            i = route[idx]
            j = route[(idx + 1) % len(route)]
            since_last_prime += 1
            if self.distance_matrix[min(i, j), max(i, j)]:
                dist = self.distance_matrix[min(i, j), max(i, j)]
            else:
                dist = City.dist(self.cities[i], self.cities[j])
            score += dist * (1 if self.cities[i].is_prime or since_last_prime % 10 else 1.1)
        return score

    def build_regions(self, max_cities: int):
        start_time = time()
        max_x = 0
        max_y = 0
        min_x = 100
        min_y = 100
        for city in self.cities:
            max_x = max(max_x, city.x)
            max_y = max(max_y, city.y)
            min_x = min(min_x, city.x)
            min_y = min(min_y, city.y)
        self.regions = [Region((min_x, min_y), (max_x, max_y), self.cities)]
        while True:
            keep = []
            new = []
            for _id, region in list(enumerate(self.regions)):
                if len(region.cities) > max_cities:
                    logging.debug(f"splitting region containing {len(region.cities)}")
                    new.extend((region.split()))
                else:
                    keep.append(region)
            if len(new) != 0:
                logging.info(f"Dropping {len(self.regions) - len(keep)} regions")
                logging.info(f"Adding {len(new)} regions")
                self.regions = keep + new
            else:
                logging.info(f"found {len(self.regions)} regions in {time() - start_time} seconds")
                return len(self.regions)

    def build_distance_matrix(self):
        start_time = time()
        self.distance_matrix = lil_matrix((self.n_cities, self.n_cities))
        for i, region_a in enumerate(self.regions):
            cities = region_a.cities
            n_neighbours = 0
            for region_b in self.regions[(i + 1):]:
                if Region.is_neighbour(region_a, region_b):
                    n_neighbours += 1
                    cities += region_b.cities
            logging.debug(f"{n_neighbours} neighbours found")
            logging.debug(f"Calculating distances for {len(cities)} cities")
            for a, city_a in enumerate(cities):
                for city_b in cities[(a + 1):]:
                    if not self.distance_matrix[min(city_a.id, city_b.id), max(city_a.id, city_b.id)]:
                        self.distance_matrix[min(city_a.id, city_b.id), max(city_a.id, city_b.id)] = City.dist(city_a,
                                                                                                               city_b)
        self.distance_matrix = self.distance_matrix.tocsr()
        save_npz('../store/dist_matrix.matrix', self.distance_matrix)
        logging.info(
            f"finished distance matrix with {self.distance_matrix.getnnz()} entries in {start_time - time()} seconds\n" +
            f" sparsity = {100 * (1 - self.distance_matrix.getnnz() / self.n_cities / self.n_cities)} percent")


# class AntColony:
#     def __init__(self, city_map: CityMap):


def main():
    c_map = CityMap(max_cities=5)

    return


if __name__ == "__main__":
    main()
