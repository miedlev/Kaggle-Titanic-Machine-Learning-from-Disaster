### Kaggle Competition | [Titanic Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview/description)

This is the legendary Titanic ML competition / the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.

#### 1. My Conclusion Analysis Report - Jupyter Notebook
* [TitanicFinal Analysis](https://github.com/miedlev/Kaggle-Titanic-Machine-Learning-from-Disaster/blob/master/TitanicFinal%20part.ipynb)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.


#### 2. Process Introduction :
It is a competition that can be said to be Kaggle's introductory period and conducts a Python-based analysis. My focusins was on 
1. dummy variable generation
2. feature engineering
3. word-based column generation
4. continuous variables range
5. tree-based model

#### 2. Dependencies & Tech :
* [NumPy](http://www.numpy.org/)
* [IPython](http://ipython.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Matplotlib](http://matplotlib.org/)
* [StatsModels](http://statsmodels.sourceforge.net/)


#### 4. Titanic Machine Learning from Disaster
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

