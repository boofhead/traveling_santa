import logging
from multiprocessing.pool import Pool
from os.path import isfile
from time import time

from numpy import nonzero, squeeze, inf, arange, zeros, zeros_like, nan, ones, in1d, nansum
from numpy.random import choice, seed
from scipy.sparse import save_npz, load_npz

from src.traveling_santa import CityMap, primes_sieve

N_CPU = 4
logging.basicConfig(level=logging.INFO)


class AntColony:
    def __init__(self, city_map: CityMap, bad_score=10 * 1515696.0, q=1.0e10, alpha=1.0, beta=2.0, ro=0.9):
        self.city_map = city_map
        self.distance_matrix = self.city_map.distance_matrix
        self.pow_dist_matrix = self.distance_matrix.power(-beta)
        self.bad_score = bad_score
        self.q = q
        self.tau = self.q / self.city_map.n_cities / self.bad_score
        if isfile('../store/pheromone_matrix.matrix.npz'):
            self.pheromone_matrix = load_npz('../store/pheromone_matrix.matrix.npz').tocsr()
        else:
            self.pheromone_matrix = (self.distance_matrix != 0).astype('float') * self.tau
        self.alpha = alpha
        self.beta = beta
        self.ro = ro

    def find_route(self, n_ants=8, n_iterations=10000):
        best_score = inf
        for _ in range(n_iterations):
            with Pool(N_CPU) as p:
                res = p.map(AntColony.generate_route,
                            [(self.pow_dist_matrix, self.pheromone_matrix, self.beta, self.alpha) for _ in
                             range(n_ants)])
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
                        if scores[_id] != nan:
                            min_id = min(route[_id], route[(_id + 1) % len(route)])
                            max_id = max(route[_id], route[(_id + 1) % len(route)])
                            assert self.pheromone_matrix[min_id, max_id], 'wrong pheromone assignment'
                            self.pheromone_matrix[min_id, max_id] += self.q / scores[_id]
                            self.pheromone_matrix[max_id, min_id] += self.q / scores[_id]
            if len(best_scores):
                best_score = min(best_scores)
            best_route = best_routes[best_scores.index(best_score)]
            with open(f"../res/aco_{int(best_score * 100000)}.csv") as f:
                f.write("Path\n")
            f.writelines([str(id_) for id_ in best_route])
            logging.info(f"updating best score : {best_score}")
            save_npz('../store/pheromone_matrix.matrix', self.pheromone_matrix)

    @staticmethod
    def generate_route(inputs):
        pow_distance_matrix = inputs[0]
        pheromone_matrix = inputs[1]
        beta = inputs[2]
        alpha = inputs[3]
        seed()
        n_cities = pow_distance_matrix.shape[0]
        route = zeros(n_cities, dtype=int)
        n = arange(n_cities, dtype=int)
        primes = primes_sieve(n_cities)
        prime_factors = 1 + (1.1 ** (-beta) - 1) * in1d(n, [0] + primes)
        not_visited = ones(n_cities, dtype=bool)
        current_city_id = 0
        scores = zeros_like(route, dtype=float)
        steps_from_last_prime = 0
        then = time()
        for i in range(n_cities):
            if not i % 1000:
                now = time()
                logging.info(f"step {i} time: {now - then}")
                then = now
            route[i] = current_city_id
            not_visited[current_city_id] = False
            pow_distances = squeeze(pow_distance_matrix.getrow(current_city_id).toarray())
            candidates = nonzero(pow_distances * not_visited)[0]
            if i == (n_cities - 1):  #
                logging.info('all cities visited')
                if 0 in nonzero(pow_distances)[0]:  # can get back to start
                    scores[i] = pow_distances[0] * (
                        1.0 if (steps_from_last_prime % 10) else 1.1)
                else:  # can't get back so jump
                    logging.info('jumping to home')
                    scores[i] = nan
                scores = scores ** -(1 / beta)
                logging.info(f"path found, approx score: {nansum(scores)}")
                return route, scores
            elif len(candidates) == 0:  # jump to unvisited and carry on
                logging.debug('ant is stuck, making random jump')
                next_id = choice(n[not_visited])
                scores[i] = nan
            else:
                pow_distances *= not_visited
                pheromones = squeeze(pheromone_matrix.getrow(current_city_id).toarray())
                if not steps_from_last_prime % 10:
                    pow_distances *= prime_factors
                probabilities = ((pheromones ** alpha) * pow_distances).take(candidates)
                probabilities /= sum(probabilities)
                next_id = choice(candidates, p=probabilities)
                scores[i] = pow_distances[next_id]
            current_city_id = next_id
            steps_from_last_prime += 1
            if current_city_id in primes:
                steps_from_last_prime = 0


def main():
    aco = AntColony(CityMap(max_cities=1))
    # aco.generate_route([aco.pow_dist_matrix, aco.pheromone_matrix, 100, aco.beta, aco.alpha])
    aco.find_route()

    return


if __name__ == "__main__":
    main()
