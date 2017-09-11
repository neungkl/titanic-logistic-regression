Titanic Logistic Regression
===

> This is homework for 2110597 Pattern Recognition class

# Problem

An implementation of logistic regression (without any machine learning library) to classify
[Titanic task](https://www.kaggle.com/c/titanic) in Kaggle competitions.

The aiming of the task is predict who is survived in Titanic sinking in 1912. Given the dataset
of crew with 891 people that labelled as survived or died, and you have to predict another 418 people with
no label.

The task is provided by Kaggle website. See more description and fully task on [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

# Usage

I recommend you to install jupyter notebook to run the code. See installation in [http://jupyter.org/install.html](http://jupyter.org/install.html). After you have completed installation, use your terminal to redirect to your
project directory, and type following command below.

```
jupyter notebook
```

Then it will popup jupyter website that you can play around with my code. If website is not appeared, you can
directly copy URL from terminal and paste to your preferable browser.

I've also provided `titanic-classification.py` for just simple run with Python. Feel free to run it and
see through it any time, this is more convenient to inspect only the code part. 

# Result

I use 4 features of dataset to train on model, which is Pclass, Sex, Age and Embarked, and train on
logistic regression model.

The model can archieved accuracy with 0.7799 score in Kaggle scoreboard which is almost the same
accuracy as validation dataset (with 0.7 split ratio on training dataset).

Here a graph of loss function (cross-entropy loss) and accuracy percentage with epoch times on x-axis.

<img width="400" alt="result" src="result.png">

# License

[MIT](LICENSE) © Kosate Limpongsa

