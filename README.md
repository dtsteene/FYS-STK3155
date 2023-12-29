# FYS-STK3155 - Applied Data Analysis and Machine Learning

A project based course at the Univeristy of Oslo. Thaught by Morten Hjorth-Jensen autumn 2023.

The course covers essential algorithms and methods for statistical data analysis and machine learning. We got to explore introductory level research problems and convey modern scientific results. It emphasizes the development of complex code, familiarity with data facilities, and skills in managing large scientific projects.

Each project folder contains a scientific report and source code, which formed the basis for grading. The raport needed to adhere to a traditional structure of abstract-introduction-theory-method-results-discussion and finaly conclusion. Method is essentialy the soruce code implementation, which natrually had the greatest weight in grading of all the sections (30%). The raports are rather lengty, respectively being 41, 61 and 35 pages long. Luckily these where group projects and I had my good friends August Femthjell and Philip Hoel on my team. Thank you for the plesant colaboration :)

## Project 1 - linear regression and resampeling techniques (grade: 95/100)
 ### Abstract
 In this project, we aim to employ various linear regression methods, including Ordinary Least Squares, Ridge, and Lasso, to analyze a set of terrain
data and Franke’s function. We explore the use of two-variable polynomials of varying orders and optimize hyperparameters for Ridge and Lasso.
To assess the effectiveness of the models, we utilize resampling techniques
such as non-parametric bootstrap and k-fold cross-validation. These techniques allow us to evaluate each model’s performance by dividing the data
into test and training subsets and training the model multiple times on
the training subset while evaluating it on the testing subset. Although
our ultimate goal is to identify the best model and evaluate its performance, we also explore and compare several regression techniques to gain
valuable insights into their strengths and weaknesses. We found our best
results using Lasso regression.
## Project 2 - Classification and Regression (grade: 100/100)
### Abstract
In this project, our focus was centered on classification and regression
problems and how to solve them with neural networks. Specifically, we
have looked at feed forward neural networks and how the optimization
works. The optimization was done through the means of the gradient
descent methods. To further optimize, we have tested different variations
of the gradient descent method, such as momentum, AdaGrad, RMSProp
and ADAM. All methods tested for both stochastic and deterministic
versions. Further, we have applied these methods to a simple univarite
polynomial, Franke’s function and the Wisconsin Breast Cancer Data.
Testing the regression methods on the univariate polynomial and Franke
function and the classification methods on the Wisconsin Breast Cancer
data set. Across the board, Adam was the top performer


## project 3 - PINNs and Explicit Forward Euler (grade: ungraded/100)
### Absract
In this project, we aimed to solve the one-dimensional heat equation using two numerical methods: the Forward Euler method and Physics-Informed Neural Networks (PINNs). We sought to draw a comparison between these methods, alongside the analytical solution. Our goal was to evaluate the performance of both PINNs and the Forward Euler method in terms of accuracy, computational speed, and the complexity of implementation. To optimize the PINNs, we experimented with various activation functions and adjusted the number of layers and nodes per layer. Concurrently, for the Forward Euler method, we manipulated different spatial steps to assess its effectiveness. At the end of the project, we observed that while PINNs may not provide a significant advantage for simpler problems, they hold potential to be more effective when dealing with more complex problems.

