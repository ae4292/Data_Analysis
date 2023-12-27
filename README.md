# Shoe-Sales-Profitability
IEOR 242 Project Proposal Group 2
By: Adrian Enders, Ameesha Khan, Rahul Ranganathan, Priyanka Ravi, Alamdeep Sethi, .Megha, Hyunjoon Kweon

# Problem
For our Project, we would like to create a classification model to predict the profitability of a new line of shoes. We want to be able to say with some level of certainty whether or not the type of shoe we are creating will be a profitable venture. Rather than a regression model, we believe that a classification model will be a more appropriate model as predicting profit margins would be difficult to do with great accuracy. By simplifying the problem to a classification problem, we will be able to give more valuable insights into new shoe production ideas.

# Data
For the data, we are going to look at Ebay’s Terapeak data. This data includes access to years of real-world sales data for millions of items. For example, The number of listings and items sold for particular items, average sales prices, the conditions of items sold, sell-through rates, sales trends over time, and unsold inventory. From here we can use sales data and other cosmetic features of the shoes to predict its profitability. While eBay is an after-market marketplace, we believe that it is a good alternative to actual sales data that companies are not likely to share.
Due to the amount of shoes we will have to collect data on, there will be lots of data cleaning from the data we are downloading. The main point of concern will be finding an easy way to aggregate the data into one data table.
https://www.ebay.com/help/selling/selling-tools/terapeak-research?id=4853

# Techniques
## Models
Logistic Regression: Scikit-Learn  
Random-Forests: Scikit-Learn  
Boosting:  XGBoostClassifier  
Validation  
K-Fold Cross Validation  
Bootstrap Aggregation  
### Evaluation 
i. Accuracy  
ii. Precision  
iii. F1 Score  
iv.  Recall  

# Impact / Goal
The impact of this project is to give shoe manufacturers another way to assess new shoes that they would like to put on the market based on previous shoe sales. This gives an empirical way to assess the possible profitability of a new shoe line which can help boost the profitability of the company.
This process can also be applied to many different kinds of consumer products. It would be helpful to many consumer brands to be able to predict the possible profitability of new products based on past data with greater efficiency.
Resellers could use such models to see what products they could resell on eBay as well

# Risk Analysis
a. Data quality - The accuracy and completeness of eBay Terapeak data might be questionable. It might contain errors or missing values which could lead to imprecise predictions.
Although this risk can be mitigated by implementing rigorous data validation, imputation, and monitoring.  
b. Model fitting- There might be a problem with overfitting or underfitting when models like logistic regression, random forest, and XGBoostClassifier are used. It may become either too complex or simple.	
Employing feature selection techniques and hyperparameter tuning can help focus on the important variables and find the right balance between complexity and performance.

