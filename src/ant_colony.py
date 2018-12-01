import logging
from multiprocessing.pool import Pool
from os.path import isfile
from time import time

from numpy import nonzero, squeeze, inf, arange, zeros, zeros_like, nan, ones, in1d, nansum, isnan
from numpy.random import choice, seed
from scipy.sparse import save_npz, load_npz

from src.traveling_santa import CityMap, primes_sieve

N_CPU = 4
logging.basicConfig(level=logging.INFO)


class AntColony:
    def __init__(self, city_map: CityMap, bad_score=10 * 1515696.0, q=1., alpha=2.0, beta=0.5, ro=0.9):
        self.city_map = city_map
        self.distance_matrix = self.city_map.distance_matrix
        self.pow_dist_matrix = self.distance_matrix.power(-beta)
        self.bad_score = bad_score
        self.q = q
        self.tau = self.q * self.bad_score / self.city_map.n_cities
        if isfile('../store/pheromone_matrix.matrix.npz'):
            self.pheromone_matrix = load_npz('../store/pheromone_matrix.matrix.npz').tocsr()
        else:
            self.pheromone_matrix = (self.distance_matrix != 0).astype('float') * self.tau
        self.alpha = alpha
        self.beta = beta
        self.ro = ro
        self.prime_factor = 1.1 ** (-self.beta)

    def find_route(self, n_ants=4, n_iterations=10000):
        best_score = inf
        for _ in range(n_iterations):
            with Pool(N_CPU) as p:
                res = p.map(AntColony.run_ant, [self] * n_ants)
            self.pheromone_matrix *= self.ro
            best_scores = []
            best_routes = []
            for route, scores in res:
                score = self.city_map.score_route(route)
                if score < best_score:
                    best_scores.append(score)
                    best_routes.append(route)
                    logging.info(f"route found through all cities, score: {score}")
                    for _id in range(len(route)):
                        if not isnan(scores[_id]):
                            min_id = min(route[_id], route[(_id + 1) % len(route)])
                            max_id = max(route[_id], route[(_id + 1) % len(route)])
                            assert self.pheromone_matrix[min_id, max_id], f"tried to assign {scores[_id]} to pheromone matrix {(min_id, max_id)}"
                            self.pheromone_matrix[min_id, max_id] += self.q / scores[_id]
                            self.pheromone_matrix[max_id, min_id] += self.q / scores[_id]
            if len(best_scores):
                best_score = min(best_scores)
                best_route = best_routes[best_scores.index(best_score)]
                save_npz('../store/pheromone_matrix.matrix', self.pheromone_matrix)
                with open(f"../res/aco_{int(best_score * 100000)}.csv", mode='w') as f:
                    f.write("Path\n")
                    f.writelines([str(id_) + '\n' for id_ in best_route])
                    f.write("0\n")
                logging.info(f"updating best score : {best_score}")

    def generate_route(self):
        seed()
        route = zeros(self.city_map.n_cities, dtype=int)
        n = arange(self.city_map.n_cities, dtype=int)
        not_visited = ones(self.city_map.n_cities, dtype=bool)
        current_city_id = 0
        scores = zeros_like(route, dtype=float)
        then = time()
        for i in range(self.city_map.n_cities):
            if not i % 1000:
                now = time()
                logging.info(f"step {i} time: {now - then}")
                then = now
            route[i] = current_city_id
            not_visited[current_city_id] = False
            pow_distances = squeeze(self.pow_dist_matrix.getrow(current_city_id).toarray())
            candidates = nonzero(pow_distances * not_visited)[0]
            if i == (self.city_map.n_cities - 1):  #
                logging.info('all cities visited')
                if 0 in nonzero(pow_distances)[0]:  # can get back to start
                    scores[i] = pow_distances[0]
                else:  # can't get back so jump
                    logging.info('jumping to home')
                    scores[i] = nan
                scores = scores ** -(1 / self.beta)
                logging.info(f"path found, approx score: {nansum(scores)}")
                return route, scores
            elif len(candidates) == 0:  # jump to unvisited and carry on
                logging.debug('ant is stuck, making random jump')
                next_id = choice(n.compress(not_visited))
                scores[i] = nan
            else:
                pow_distances *= not_visited
                pheromones = squeeze(self.pheromone_matrix.getrow(current_city_id).toarray())
                if not ((current_city_id in self.city_map.primes) or (i+1) % 10):
                    pow_distances *= self.prime_factor
                probabilities = ((pheromones ** self.alpha) * pow_distances).take(candidates)
                probabilities /= sum(probabilities)
                next_id = choice(candidates, p=probabilities)
                scores[i] = pow_distances[next_id]
            current_city_id = next_id

    @staticmethod
    def run_ant(aco):
        return aco.generate_route()


def main():
    aco = AntColony(CityMap(max_cities=1))
    # aco.generate_route()
    aco.find_route()

    return


if __name__ == "__main__":
    main()
