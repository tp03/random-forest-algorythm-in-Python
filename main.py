from AlgorythmRunner import run_made_algorythm, run_sklearn_algorythms
import random

iterations = 25
trees = 30
type = "entropy"
data = "breast-cancer"
seeds = [random.randint(1, 100) for i in range(iterations)]
run_made_algorythm(data, iterations, trees, seeds, type)
run_sklearn_algorythms(data, iterations, trees, seeds, type)
