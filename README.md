# Likert-estimation
This estimator is used with survey data measured using Likert scales as the dependent variable

## Example
```python
from SurvivalScale import get_results
# data is a pandas DataFrame containing your survey data
cj, beta, metrics = get_results(data, bootstrap_iterations=5234, alpha=0.05, columns=['feature1', 'feature2'], block_id='block')
```

