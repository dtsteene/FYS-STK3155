# Regression analysis and resampling methods applied to Franke’s Function and Topographical data

## Abstract 
In this project, we aim to employ various linear regression methods, including Ordinary Least Squares, Ridge, and Lasso, to analyze a set of terrain
data and Franke’s function. We explore the use of two-variable polynomials of varying orders and optimize hyperparameters for Ridge and Lasso.
To assess the effectiveness of the models, we utilize resampling techniques
such as non-parametric bootstrap and k-fold cross-validation. These techniques allow us to evaluate each model’s performance by dividing the data
into test and training subsets and training the model multiple times on
the training subset while evaluating it on the testing subset. Although
our ultimate goal is to identify the best model and evaluate its performance, we also explore and compare several regression techniques to gain
valuable insights into their strengths and weaknesses. We found our best
results using Lasso regression.
