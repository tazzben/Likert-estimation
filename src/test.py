import pandas as pd
import SurvivalScale as s


def testFunction():
    data = pd.DataFrame({
        'k': [0, 1, 2, 3, 4, 0, 1, 2, 8, 9],
        'question': ['q1', 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q2', 'q2', 'q2'],
        'bound': [4, 4, 4, 4, 4, 9, 9, 9, 9, 9],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })
    cj, beta = s.get_results(data, columns=['feature1', 'feature2'], bootstrap_iterations=10000, alpha=0.05)
    print("Cj DataFrame:")
    print(cj)
    print("\nBeta DataFrame:")
    print(beta)


if __name__ == '__main__':
    testFunction()

