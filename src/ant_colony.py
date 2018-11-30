import logging
import random
from multiprocessing.pool import Pool
from os.path import isfile

from numpy import nonzero, squeeze, array, isin, inf
from numpy.random import choice
from scipy.sparse import save_npz, load_npz

from src.traveling_santa import CityMap

N_CPU = 4
logging.basicConfig(level=logging.INFO)


class AntColony:
    def __init__(self, city_map: CityMap, bad_score=10 * 1515696.0, q=1.0, alpha=1.0, beta=2.0, ro=0.9):
        self.city_map = city_map
        self.city_map.distance_matrix = self.city_map.distance_matrix
        self.bad_score = bad_score
        self.q = q
        self.tau = self.q / self.city_map.n_cities / self.bad_score
        if isfile('../store/pheromone_matrix.matrix.npz'):
            self.pheromone_matrix = load_npz('../store/pheromone_matrix.matrix.npz').tocsr()
        else:
            self.pheromone_matrix = (self.city_map.distance_matrix != 0).astype('float') * self.tau
        self.alpha = alpha
        self.beta = beta
        self.ro = ro

    def find_route(self, n_ants=32, n_iterations=10000):
        best_score = inf
        for _ in range(n_iterations):
            with Pool(N_CPU) as p:
                res = p.map(AntColony.run_ant, [self] * n_ants)
            self.pheromone_matrix *= self.ro
            best_scores = []
            best_routes = []
            for route, scores, score in res:
                if score < best_score:
                    best_scores.append(score)
                    best_routes.append(route)
                    logging.info(f"route found through all cities, score: {score}")
                    for _id in range(len(route)):
                        min_id = min(route[_id], route[(_id + 1) % len(route)])
                        max_id = max(route[_id], route[(_id + 1) % len(route)])
                        if self.pheromone_matrix[min_id, max_id]:
                            self.pheromone_matrix[min_id, max_id] += self.q / scores[_id]
            if len(best_scores):
                best_score = min(best_scores)
                best_route = best_routes[best_scores.index(best_score)]
                with open(f"../res/aco_{int(best_score * 100000)}.csv") as f:
                    f.write("Path\n")
                    f.writelines([str(id_) for id_ in best_route])
                logging.info(f"latest best score : {best_score}")
                save_npz('../store/pheromone_matrix.matrix', self.pheromone_matrix)

    def generate_route(self):
        route = []
        starting_id = 0  # random.choice(range(self.city_map.n_cities))
        current_city = self.city_map.cities[starting_id]
        scores = []
        steps_from_last_prime = 0
        while True:
            route.append(current_city.id)
            distances = squeeze(array(self.city_map.distance_matrix[current_city.id, :].todense())) + squeeze(
                array(self.city_map.distance_matrix[:, current_city.id].todense()))
            pheromones = squeeze(array(self.pheromone_matrix[current_city.id, :].todense())) + squeeze(
                array(self.pheromone_matrix[:, current_city.id].todense()))
            candidates = nonzero(distances)[0]
            candidates = candidates[isin(candidates, route, invert=True)]
            if len(candidates) == 0:
                if len(scores) == (self.city_map.n_cities - 1):  # all cities visited
                    if starting_id in nonzero(distances)[0]:  # can get back to start
                        scores.append(distances[starting_id] * (
                            1.0 if (steps_from_last_prime % 10 or self.city_map.cities[starting_id].isprime) else 1.1))
                    else:  # can't get back so jump
                        scores.append(None)
                    score = self.city_map.score_route(route)
                    return route, scores, score
                else:  # jump to unvisited and carry on
                    next_id = random.choice([x for x in range(self.city_map.n_cities) if x not in route])
                    scores.append(None)
            else:
                distances = distances[candidates]
                pheromones = pheromones[candidates]
                distances = distances * (1.0 if steps_from_last_prime % 10 else 1.1)
                probabilities = (pheromones ** self.alpha) / (distances ** self.beta)
                next_id = choice(candidates, p=probabilities / sum(probabilities))
                scores.append(distances[candidates == next_id][0])
            current_city = self.city_map.cities[next_id]
            steps_from_last_prime += 1
            if current_city.is_prime or not current_city.id:
                steps_from_last_prime = 0

    @staticmethod
    def run_ant(aco: "AntColony"):
        return aco.generate_route()


def main():
    aco = AntColony(CityMap(max_cities=1))
    # aco.generate_route()
    aco.find_route()

    return


if __name__ == "__main__":
    main()
