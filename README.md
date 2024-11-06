# Introduction
<b>This project was assigned as a final project for the Machine Learning course as part of my B.Sc. in the field of computer science at Tel Aviv University, and was done in pairs.
The project was implemented using the Anaconda Python package in a Jupyter notebook, using libraries we became familiar with during the course, such as NumPy, Pandas, Seaborn, etc.</b>

The dataset provided to us for this project contained data on user sessions on an online shopping website. Each session can end with a purchase or without a purchase.
The project dealt with a binary classification problem, where we classified records into two categories: 
1 for users who made a purchase and 0 for users who did not make a purchase. We relied on 23 features for the project, some of which are known and some are unknown. 

The project is divided into 3 main phases: 
1. Data exploration and processing – In this section, we examined each feature in depth and made appropriate changes to the entire database to support our classification, including dimensionality reduction.
2. Building and running the models – we chose 4 different models that were taught in class and evaluated their performance.
3. Performing the prediction using the best model – we used the best model we created to predict the likelihood of purchase on the new database, where there is no prior knowledge of whether the buyer purchased or not.

# Project steps

### Data investigation:

Initially, we examined all 23 columns in the database: we checked how many empty values each variable has, the distributions of the variables (with an emphasis on checking for outliers), 
and looked for correlations between the different variables. 

We received several different types of data: Numerical data – discrete or continuous.
Categorical variables – binary categories, Boolean categories, verbal categories, and numerical categories.

We examined each variable separately while making the necessary adjustments for further processing.

### Data processing:
We processed the data by column. The transformation actions we performed: we filled in missing values, converted values to numerical, 
aggregated information (for example, quarterly information), removed outliers, reduced dimensions, and deleted some variables due to negligible correlation with the label or 
because of a significant amount of missing data.

### Building machine learning models:
We chose to use the following models: Naive Bayes classifier, Logistic regression, Random Decision tree, RandomForest classifier.

We used Grid Search to select the best hyperparameters. We divided the processed database into train and validation.
We relied on the training set to find the best hyperparameters through Grid Search, and with the validation set, we examined our results.
Since there is a need to maximize the AUC of each model, we used Grid Search to maximize the AUC for the data.
Between model and model, we examined various parameters that we believe are significant and influential for the tested model.

### Model evaluation:
Building the Confusion Matrix – this is the matrix for the RandomForest Classifier model
<br><br>
<div align="center">
<img src="https://github.com/user-attachments/assets/c2bac0e7-f633-430b-b080-bb5bc7190d50" class="center">
</div>

The X-axis represents what actually happened: whether a purchase was made or not. The Y-axis represents the predictions of the model: whether the model predicted that a purchase would occur or not.

<br>In the matrix, the bottom right value is TP: the number of times the model correctly predicted a purchase. 
<br>The bottom left value is FP: the number of times the model incorrectly predicted that there was a purchase.
<br>The top right is FN: the number of times the model incorrectly predicted that there was no purchase.
<br>The top left value is TN: the number of times the model correctly predicted that there was no purchase.  

Our model predicted correctly more times than not, with less than 5.5% of the predictions being incorrect.

The evaluation of the models was done using K-Fold, with the KfoldPlot function. We ran this on the validation set, thus we evaluated our models. 
We plotted ROC curves and calculated the AUC for each model for the train and validation sets. 
<div align="center">
<img src="https://github.com/user-attachments/assets/ec75a8e8-a681-44d6-8509-57549d494bac" class="center">
</div>
<div align="center">
<img src="https://github.com/user-attachments/assets/0d2e5b1b-ae80-40c4-ab7b-6fc29ebf0f26" class="center">
</div>
<br>
In light of our results, we found that the RandomForest Classifier model has the highest AUC for both validation and train, and therefore we chose this model as our predictive model.

### Execution of the prediction:
We read the test file and ran the entire processing procedure on it that was done on the train file. 
We ran the RandomForest Classifier model again on all the values from the train set after processing.  
After that, we performed the prediction on the test file and received the probability of making a purchase or not for each individual ID. 
Finally, we put the results into one file. Additionally, at the end of the program, we added a pipeline. In other words, a piece of code that briefly performs the entire process mentioned above.

# Summary
We worked on a project involving a binary classification problem in the field of online shopping. 
We used various tools and applied the course material practically. The main goal of the project was to maximize the AUC in order to predict whether a purchase was made, in the best possible way.
We conducted an investigation and data processing in the project, from which we learned about the variables and drew conclusions, 
addressed missing and outlier data, and worked to reduce dimensions using various techniques. 
After that, we built machine learning models and ran them on the dataset. 
After evaluating the models using K-Fold, we found that the model with the highest AUC score was the RandomForest Classifier. 
We decided to perform the prediction using this model on the test file and saved the results in a new file. 


In summary, the project was challenging but successful. We invested a lot of time, thought, and effort to get the most out of every step we took together in the project.
We met regularly to work on the project together from start to finish and delved into the smallest details. We were partners in every part of the process, we learned a lot, and we grew together.
