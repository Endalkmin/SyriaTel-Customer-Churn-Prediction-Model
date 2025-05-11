# SyriaTel-Customer-Churn-Prediction-Model
**Author**: Endalkachew Dessalegne

## Project Overview
This project focuses on building a classifier model to predict whether a customer will ("soon") stop doing business with SyriaTel, a telecommunications company. The company is interested in reducing the money lost because of customers who don't stick around very long. Which factors contribute most to customers churn? are there any predictable patterns in the dataset? which model can better predict customers who will churn? What correctve action can the company take to reduce number of customers who churn?

## Business Understanding 
 ## Business Problem 
 ### Key Business Questions 
* which model can better predict customers who will churn?
* can this predictive model improved?
* which factors contribute most to customers churn?
* are there any predictable patterns in the dataset?
* What correctve action can the company take to reduce number of customers who churn?
  
***
## Data
The data for this project came from the website [kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset). 
The dataset consists of information about SyriaTel customers. The dataset has 3333 entries of cutomers and 21 columns or features. Each entry gives information related to a customers use of the services related to the telecommunication company. This features include account length, the duration of call a customer made during the day, evening and night and the amount charged. It also include wheather a customer uses international calls and voice mail plan or not.  
*** 

After checking the information on the dataset to see column names, number of column and rows, and checked the data types I identified the target variable. The target variable is the column 'churn', asigned to a variable y, and its value count shows it has two classes, 'True' and 'False'. All the other independent variables are selected in a dataframe and assigned a variable X. This indicates it is a categorical data.
Before preprocessing was done on the dataset, I splited the data into train and test set.
I then cheked the data for missing values and found out that there are no missing values. 
Then I droped columns or features which are not relevant for this project. These include 'state', 'area code' and 'phone number'.
Then I selected 'object' data types and assigned to them to them a variable 'X_train_categorical'. This datafram is then delt with to convert their categorical variable, 'yes' and 'no' into dummy encoded variables. 
***
Then numerical variables in the X_train_numerical, which are not on the same scale, are normalized with MinmaxScalar to a consistent scale of 0 to 1.
Then the X_train_encoded and X_train_scaled are concatinated together. This is assigned to a variable 
X_train_all. 

***
## Modeling 
Since the target variable is a category a classification model is used for prediction.
As a baseline model, I choose Logistic Regression to fit the preprocessed training data set and trained.

## Model Evaluation 
The performance of the baseline model is evaluated using both the train and test data set. Since the company is interested in identifying customers who will churn (True postive), finding a model with high precision(higher percentage of the true positive values) is more important. Thus, in evaluation and comparison of performance of models, I will use precision of models. Then I will optimize the model to look for a model with high precision. For the company even if correctly predicting customers who are going to churn(True postive) is very crucial. High precision is useful in this case because we need to minimize false positives (wrong positive classifications) which may result in loss of loyal customers. It is better to have false negatives than to have false positive and lose a customer
That precission is used as the factor for evaluating performance of the model.

The evaluation of performance of the baseline model is done on training and testing data. 
For iteration of model performance and to find the evaluation matrics for baselin and other models I used two functions, evaluation_metrix and metrix_summary. 
The testing data set is also preprocessed. 
The performance of the baseline model is low, with precision of only 50% on both training and testing data set. This means only 50% of the customers who will churn will be correctly predicted.
The accuracy of the model is better, but it is misleading because the dataset is imbalanced.
![img](images/Conf_matrix_LR.png) 
The recall-score is also very low which indicates that lower percentage actual positives were correctly identified.
Thus, I checked if the precision of the model improves by addressing the imbalance in the dataset.

*** 
SMOTE (Synthetic Minority Over-sampling Technique) method is used to handle imbalanced datasets by generating synthetic samples for the minority class, in this case class 1 or 'True'. Precision of updated model on train data set has improved but its precision on test set of data, which reflects real-world performance on imbalanced data, is low(0.27) compared to the baseline model. This is because of overfitting of the model. Thus, we have to find a new model which with better precision.
DecisionTreeClassifier will be the the new model to train.
![img](images/conf_matrix_LR_SMOTE.png) 
***
DecisionTree model is trained with the same preprocessed data and used to predict on the train and test sets. 

Compared to Logestic regression, the performance of the DecisionTree model shows improvement. Precision of the model on the train data set is 100% but it is 64% on the test data set. This is a significant change. There is overfitting of the model. Next, the performance of Decision Tree model will be optimized by selecting the best values for parameters like max_depth, min_samples_split, and criterion. 
![img](images/conf_matrix_dtree.png)
Hyperparameter Tuning was done for DecisionTree Model and using the bast parameters the model was trained. Precision of the optimized Decision Tree model, on test set, has increased to 84%. Its precision on predicting the "False" or (0) class correctly is very high 96%. This helps to avoid misclassifying customers who are not going to churn. But still, about 85% precision is not enough. It has to be improved. A better model Random Forest will be trained. 
![img](images/conf_matrix_dt_best.png)
***
A RandomForest model was trained with the preprocessd data and used to predict on the train and test sets. Precision of the model on test dataset is now 89.12% but lower than that of train dataset. This shows better performance from previous models. The accuracy of the data is now very high, 96%. Still there is overfitting of the model. For optimaized performance we will find best parameters for the random forest model and run it again.
![img](images/conf_matrix_rf.png) 
Hyperparameter Tuning was done for Random Forest Model and using the bast parameters the model was trained. This model has high precision of 90% which show improvment from the previous models. Its accuracy is also high. Its precision on predicting the "False" or (0) class correctly is very high 97%. 
![img](images/best_conf_matrix_rf.png)
*** 
It is very important to find out which features are curucial to indicate which customers are to churn and which ones are to stay. Thus importance scores shows that Total day minutes 'Total day charge', 'Customer service calls', 'International plan', and 'Total eve charge' are among the top featues which are strong predictors. 
![img](images/feature_import.png)
 ## Conclusion 
 * Precision is the best evaluation metrics for model performance to maximize the correct prediction of coustomers who will churn
* RandomForestClassifier is the best performing model with the highest precision
* This model can predict 'True Postive' or customers who are going to churn with better precision and minimizes "False positves". This means has minimum chance of incorrectly identifying customer who is going to stay as a cutomer who is going to churn. 
* In trying to find the best model the precision improved from 50% of the baseline model to 90% precision of RandomForest model
* This performance improvment is achived by reducing overfitting of the models.
* To reduce the money lost because of customers who don't stick around very long, the indicators will help to act accordingly.

## Recommendations 
The telecommunication company is recommended to take actions on the following main factors which may lead to customers chrun or stay. 

To reduce customers who will churn: 

* Offer competitive pricing and bundle plans to retain customers who use more daytime and evening minutes and pay higher charges 
* Frequent 'customer service calls' indicate customer dissatisfaction which may lead to churn. THus improv service quality and responsiveness
*  Offer packages and competitive rates to customers with international plans, those who make overseas communication, for better retention.

  ***
## Repository Structure

Describe the structure of your repository and its contents, for example:


```
├── README.md                                <- The top-level README for reviewers of this project
├── phase-3-project.ipynb   <- Narrative documentation of analysis in Jupyter notebook
├── Presentation.pdf                         <- PDF version of project presentation
├── data                                     <- Both sourced externally and generated from code
└── images generated from code                        
```
