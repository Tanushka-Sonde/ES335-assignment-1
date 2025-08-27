import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree

np.random.seed(42)
num_average_time = 3    


def generate_data(N, M, input_type="real", output_type="discrete"):
    """Generate dataset with N samples, M features for given input/output types."""
    if input_type == "discrete":
        X = pd.DataFrame(np.random.randint(0, 5, size=(N, M)),
                         columns=[f"f{i}" for i in range(M)])
    else:
        X = pd.DataFrame(np.random.randn(N, M),
                         columns=[f"f{i}" for i in range(M)])

    if output_type == "discrete":
        y = pd.Series(np.random.randint(0, 2, size=N))
    else:
        y = pd.Series(np.random.randn(N))

    return X, y


def measure_runtime(N_values, M_values):
    cases = {
        "Disc-Inp Disc-Out": ("discrete", "discrete"),
        "Real-Inp Disc-Out": ("real", "discrete"),
        "Disc-Inp Real-Out": ("discrete", "real"),
        "Real-Inp Real-Out": ("real", "real"),
    }

    results = {case: {"fit_N": [], "pred_N": [],
                      "fit_M": [], "pred_M": []}
               for case in cases.keys()}

    # Vary N (fix M)
    fixed_M = 5
    for N in N_values:
        for case, (inp_type, out_type) in cases.items():
            fit_times, pred_times = [], []
            for _ in range(num_average_time):
                X, y = generate_data(N, fixed_M, inp_type, out_type)
                criterion = "entropy" if out_type == "discrete" else "mse"
                tree = DecisionTree(criterion=criterion, max_depth=5)

                t0 = time.time()
                tree.fit(X, y)
                fit_times.append(time.time() - t0)

                t0 = time.time()
                tree.predict(X)
                pred_times.append(time.time() - t0)

            results[case]["fit_N"].append(np.mean(fit_times))
            results[case]["pred_N"].append(np.mean(pred_times))

    # Vary M (fix N)
    fixed_N = 200
    for M in M_values:
        for case, (inp_type, out_type) in cases.items():
            fit_times, pred_times = [], []
            for _ in range(num_average_time):
                X, y = generate_data(fixed_N, M, inp_type, out_type)
                criterion = "entropy" if out_type == "discrete" else "mse"
                tree = DecisionTree(criterion=criterion, max_depth=5)

                t0 = time.time()
                tree.fit(X, y)
                fit_times.append(time.time() - t0)

                t0 = time.time()
                tree.predict(X)
                pred_times.append(time.time() - t0)

            results[case]["fit_M"].append(np.mean(fit_times))
            results[case]["pred_M"].append(np.mean(pred_times))

    return results


def plot_results(N_values, M_values, results):
    plt.figure(figsize=(14, 6))

    # Fit/Pred vs N
    plt.subplot(1, 2, 1)
    for case in results.keys():
        plt.plot(N_values, results[case]["fit_N"], label=f"Fit-{case}")
        plt.plot(N_values, results[case]["pred_N"], '--', label=f"Pred-{case}")
    plt.xlabel("N (samples)")
    plt.ylabel("Time (s)")
    plt.title("Runtime vs N (M fixed)")
    plt.legend()

    # Fit/Pred vs M
    plt.subplot(1, 2, 2)
    for case in results.keys():
        plt.plot(M_values, results[case]["fit_M"], label=f"Fit-{case}")
        plt.plot(M_values, results[case]["pred_M"], '--', label=f"Pred-{case}")
    plt.xlabel("M (features)")
    plt.ylabel("Time (s)")
    plt.title("Runtime vs M (N fixed)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N_values = [50, 100, 200, 400]  
    M_values = [2, 5, 10, 20]       
    results = measure_runtime(N_values, M_values)
    plot_results(N_values, M_values, results)

    print("\n=== Theoretical Complexity ===")
    print("Training: O(N * M * log N)")
    print("Prediction: O(log N) per sample")
