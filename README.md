# Estimate Likert-based Items with Survival Scale

This estimator is used with survey data measured using Likert scales as the dependent variable. It envisions the Likert scale as a series of trials where each trial has a probability of success characterized by a set of features on a logit scale. This approach allows the bottom and top box of the Likert scale to have different probabilities of success (censoring) than the middle boxes.  This model is capturing the idea that very unhappy and very happy respondents represent a much more extreme position than those in the middle of the scale.

This estimator is a expansion of the estimators described in the following papers:

Smith, B. O., & Wooten, J. J. (2023). Assessing proxies of knowledge and difficulty with rubric-based instruments. Southern Economic Journal, 90(2), 510–534. https://doi.org/10.1002/soej.12658

Smith, B. O., & Wooten, J. J., (2024 - Working Paper). Are Students Sexist when Rating Each Other? Bias in Peer Ratings and a Generalization of the Rubric-Based Estimator. Available at SSRN: https://dx.doi.org/10.2139/ssrn.4858815

These papers focus on the use of this estimator in educational settings.  This paper and software allow the bottom box to have different probabilities of success than the middle boxes thus making it more suitable for Likert scale data.  This estimator and an application is described in a upcoming paper:

Smith, B. O., Klucarova, S., & Wooten, J. J., (2025 - Working Paper) What Advertising Techniques Increase Stated Preference for Microtransit: An Application of a New Estimation Technique. 

This new estimator is not a replacement for ordered logit or probit models.  It is a different way of looking at the data that may be more suitable in some settings. 

## Example
```python
from SurvivalScale import get_results
# data is a pandas DataFrame containing your survey data
cj, beta, metrics = get_results(data, bootstrap_iterations=1000, alpha=0.05, columns=['feature1', 'feature2'], block_id='block')
```

