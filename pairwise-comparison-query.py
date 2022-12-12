import pySequentialLineSearch as pysls
import numpy as np
import sys
from typing import List


def gen_initial_query(num_dims: int, num_options: int) -> List[np.ndarray]:
    """Generate a query for the first iteration."""
    #print("strategy: ", [np.random.rand(num_dims) for i in range(num_options)], "\n")
    #return [np.random.rand(num_dims) for i in range(num_options)]
    #return [array([0.606, 0.638, 0.702]), array([0.394, 0.362, 0.298])]
    print("strategy: [0.606, 0.638, 0.702], [0.394, 0.362, 0.298]", "\n")
    return [[0.606, 0.638, 0.702], [0.394, 0.362, 0.298]]


def calc_simulated_objective_func(x: np.ndarray) -> float:
    """Calculate a synthetic objective function value."""
    return float(-np.linalg.norm(x - 0.2))


def ask_human_for_feedback(options: List[np.ndarray]) -> int:
    """Simulate human response to a pairwise comparison query."""
    assert len(options) == 2

    i_max = -1
    f_max = -sys.float_info.max

    for i in range(len(options)):
        x = options[i]
        f = calc_simulated_objective_func(x)

        if f_max < f:
            i_max = i
            f_max = f

    return i_max


def run_optimization() -> None:
    """Run optimization using preferential Bayesian optimizer."""
    #num_dims = 5
    num_dims = 3
    strategy = pysls.CurrentBestSelectionStrategy.LastSelection

    optimizer = pysls.PreferentialBayesianOptimizer(
        num_dims=num_dims,
        initial_query_generator=gen_initial_query,
        current_best_selection_strategy=strategy)

    optimizer.set_hyperparams(kernel_signal_var=0.50,
                              kernel_length_scale=0.10,
                              kernel_hyperparams_prior_var=0.10)

    print("#iter,residual,value")

    for i in range(30):
        options = optimizer.get_current_options()
        chosen_index = ask_human_for_feedback(options)
        optimizer.submit_feedback_data(chosen_index)
        optimizer.determine_next_query()

        residual = np.linalg.norm(optimizer.get_maximizer() - 0.2)
        value = calc_simulated_objective_func(optimizer.get_maximizer())

        print("{},{},{}".format(i + 1, residual, value))


if __name__ == '__main__':
    run_optimization()
