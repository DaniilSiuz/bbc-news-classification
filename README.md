# BBC news classification
Benchmarking four text classifiers on BBC news, and why cosine distance beats Euclidean in sparse high-dimensional space — Python, scikit-learn

## Which classifier reads the news best — and why?

Four classifiers on a binary news-topic task (technology vs entertainment). The interesting result isn't which one won — it's that kNN's performance was driven by the distance metric, not by k: cosine reached 0.956 mean CV macro F1 while Euclidean and Manhattan degraded sharply as k grew.

Stack: Python · scikit-learn · pandas · matplotlib

Group project, COMPSCI 361, team of 6. I owned Task 2. The analysis and write-up below reflect the full group's work.

## Setup
534 BBC news articles, split 428 train / 106 test.
Bag-of-words representation, 13,518 terms. The vectoriser is fit on the training split only, so no test vocabulary leaks into training.
Scored on macro F1, not accuracy, because the test split is imbalanced (61 entertainment / 45 technology) while the training split is roughly even.
Model selection by 5-fold stratified cross-validation over 35+ hyperparameter configurations.

## Models compared
Model	Best CV macro F1	Test macro F1
Multinomial Naive Bayes	0.986	0.981
Linear SVM	0.979	1.000
Neural network (1 hidden layer)	—	1.000
kNN (cosine)	0.956	0.952

## The finding worth reporting

For kNN, sweeping k across three distance metrics showed the metric mattered far more than the neighbourhood size. Cosine held up across all values of k; Euclidean and Manhattan fell away as k increased.

This is what you'd expect in a sparse, high-dimensional bag-of-words space. Document vectors differ enormously in magnitude simply because articles differ in length, so Euclidean distance partly measures "how long is this article" rather than "what is it about". Cosine discards magnitude and compares direction only.

Learning curves over training fractions 0.1–0.9 tell a related story: Naive Bayes, SVM and the neural net all saturate on roughly 10% of the training data. kNN was the only model still meaningfully data-limited.

## Why the perfect scores don't mean much

The SVM and the neural network both hit 1.000 macro F1 on the test set. That is a fact about the task, not a claim about the models.

With only 106 test articles, a Wilson interval around a perfect score still spans roughly [0.965, 1.000] — the data are consistent with true accuracy as low as ~96.5%. Every model clears 0.95 using a tenth of the training data. Technology and entertainment articles share very little vocabulary, so the task is close to linearly separable and the gaps between models are not resolvable at this sample size.

## Limitations
No stopword removal or TF-IDF weighting, so the raw count vocabulary is dominated by function words. TF-IDF would be the obvious next step.
Binary task only; the full BBC corpus has five categories, and the harder pairs (business vs politics) are not tested here.
106 test documents is too few to distinguish between the top three models.

## Running it

See data/README.md for the dataset source. Open analysis.ipynb and run top to bottom.
