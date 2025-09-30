import pandas as pd
import SurvivalScale as s


def testFunction():
    data = pd.DataFrame({
        'k': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        'question': [1, 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q2', 'q2', 'q2', 'q1', 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q2', 'q2', 'q2'],
        'bound': [4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 4.2, 4, 4, 4, 4, 9, 9, 9, 9, 9],
        'feature1': [1, 2, 3, 4, 0.3, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7.2, 6, 5.6, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'feature3': [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 'test', 1, 0, 0, 0, 1, 1, 1, 1, 1],
        'feature4': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        'block': [1, 1, 1, 1, 1, 2, 2, 'test', 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    })
    cj, beta, metrics = s.get_results(data, bootstrap_iterations=5234, alpha=0.05, columns=['feature1', 'feature2', 'feature3', 'feature4', 'block'], block_id='block')
    print("Cj DataFrame:")
    print(cj)
    print("\nBeta DataFrame:")
    print(beta)
    print("\nMetrics:")
    print(metrics)

if __name__ == '__main__':
    testFunction()
