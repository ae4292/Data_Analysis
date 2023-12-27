
# Abstract
The secondary market for sneakers has grown rapidly in recent years, leading it to become an important factor in the profitability of certain shoe brands. This study aims to determine with a high level of certainty whether a particular type of sneaker will be a profitable venture or not. We believe the percentage increase in a sneaker's retail price in the secondary market accurately represents a shoe's demand and popularity. Thus, we will analyze secondary markets to predict a sneaker’s relative popularity and demand through this. 
To achieve this goal, we will first collect data from secondary markets and do some feature engineering. Then, we will use various supervised and unsupervised machine learning algorithms to predict a sneaker’s price percentage increase based on the data we collected. Doing so allows us to provide brands with empirical decision-making abilities on managing inventory levels, determining optimal pricing strategies, maximizing profits, and recommending market expansion strategies. 

# Exploratory Data Analysis and Data Collection
In addition to the Stock Contest Data from 2019, sneaker data was scraped from StockX by manually recording various colors of sneakers sold in the secondary market. The final dataset included 99,956 observations across nine features. The features included Order date, Brand, Sale Price, Retail Price, Release Date, Shoe Size, Buyer Region, Colour, and Number of Sales. 
We one-hot encoded features such as color and region to provide the necessary features to regress on. The outcome variable for this study, “Price Percentage Change”, was created by dividing the difference in secondary market price and the sales price by the sneaker’s original price. This was done so that we could compare differently priced shoes. This also meant we couldn’t include secondary market prices as it would cause bias in our models.
In order to familiarize ourselves with the data, we performed extensive exploratory data analysis, including descriptive statistics and correlation analysis. Results of our findings revealed that the majority of sneakers were sold in the regions of California, New York, Florida, and Texas seen in Appendix Figure 1. When looking at price percentage increases, most values were within the range of 0.1% to 5%. The average price percentage increase was 1.24%. However, there were many outliers in the data, with some sneakers having the potential to garner up to a 12% price percentage increase, as seen in Appendix Figures 2, 3, and 4. 
An analysis of shoe brands revealed that Yeezys had the highest total number of sales. Air Jordan had the highest average price percentage change, around 4%, with some Air Jordans getting around 20% above retail price.
Lastly, K-means Clustering was performed to discover any natural groupings in the data for some form of ensemble learning. Features “days since released” and “number of sales” were chosen variables for our clustering due to their positive correlation of 0.68. The optimal number of clusters was determined by a scree plot, as seen in Figure 5, which was 4 clusters. Applying the Cluster-then-predict methodology, our linear regression model had an RMSE of 1.16 as compared to the RMSE of the baseline model of 2.26. In this case, the baseline model would predict a Price percentage increase to be the average of 1.24%. While the model predicted better than the baseline, it had a low OSR2 of 0.48. K-means clustering neither revealed any significant clusters, as seen in Appendix Figure 6, nor did it perform well on unseen data. This is because the clusters had little separation between them, thus leading to poor model results.

# Model Analysis

Linear Regression

With this model, we were able to remove features such as “Other States”, “New York”, and “Texas”. This was to remove the multicollinearity of region features and remove statistically insignificant features based on their p-value.
In addition to a normal Ordinary Least Squares regression, we performed Ridge and LASSO regression. Ridge regression provided similar accuracy, while LASSO regression had slightly lower OSR2. No notable differences from normal linear regression.
This was not a very accurate model compared to other regression models. However, it was useful for feature selection, which could be used for other models. One can also use this model to interpret the effects of certain features to help make better product decisions.

Cart Model and PCA

For our Cart Model, we used a Scit-kit Learn DecisionTreeRegressor. For parameter selection of our ccp_alpha, we found that a ccp_alpha of zero had the lowest Mean Squared Error, as seen in the graph below. This model achieved a high OSR2 of 0.978, which proved that the CART model was relatively accurate at predicting Price Percentage Change. 
We also performed a Principal Component Analysis with this model. We used Scit-Kit Learn for preprocessing and decomposition for our PCA that selects the minimum number of components such that 95% of the variance was retained. This proved quite useful as we can reduce dimensionality by almost half while achieving a similar OSR2 score.
Such a high OSR2 score may be due to the overfitting of the data. Therefore, we limited the depth of the CART model to reduce the possible overfitting. We used Scit-Kit Learn Grid-Search cross-validation, which came up with an optimal depth of 10. With this maximum tree depth, we refitted the CART model and got an OSR2 value of 0.959 with an RMSE of 0.303. Since the OSR2 value only dropped slightly compared to the previous CART model, we could conclude that a more complex model was not significantly overfitting the data and that CART was a good model for predicting Price Percentage Change. 

Time Series Analysis 

Trying to predict Sales percentage change based on a given release date proved to be difficult. However, based on certain days of the week or given month, it is easy to see some correlation between the Time of Release and Sales Percentage Change. This highlighted the importance of introducing features based on the day of the week and month. However, introducing such variables only affected the Linear Regression model’s OSR2, increasing it by roughly 0.035.

Random Forest

We used a Sci-kit Learn Random Forest Regressor for our Random Forest model. Since this is similar to a DecisionTreeRegressor, applications of PCA and parameter selection would prove to have similar results. For example, limiting the depth and leaf node sizes, the OSR2 squared only went down to 0.003. Similarly, PCA would work well with this model based on the user's needs. This proved to be our best model based on the OSR2 and RMSE.

Boosting 

Boosting was one of our strongest methods, given the RMSE and OSR2 values. Comparing the models, the XGBoost Regressor was the best for our Boosting regressor and the best model overall. Given these results, we suggest a company to use our XGBoost model for their predictions. While AdaBoost had an acceptable RMSE of 1.6, the reason the OSR2 was negative was that the model it created was most likely non-linear. This is a necessary assumption for using OSR2.

# Impact of the Results
This project gives shoe manufacturers another way to assess new shoes they would like to put on the market based on previous shoe sales. This gives an empirical way to assess the possible profitability of a new shoe line, which can help boost the company's profitability.
This process can also be applied to many different kinds of consumer products. It would be helpful to many consumer brands to be able to predict the possible profitability of new products with greater efficiency based on past data. Resellers could also use such models to see what products they could resell with greater profitability.
Given that the models are highly accurate, companies could use these results to model their products' demand and popularity properly. This could help companies better manage inventory and price their products because they can better understand their market. This not only increases efficiency it also increases profitability if used correctly.
Based on our analysis, the Random Forest Model was our best model, given our evaluation using OSR2 and RMSE. Despite this, XGBoost and our CART Model proved only marginally worse. This model could prove quite useful for shoe resellers and shoe companies as it is quite effective in predicting the performance of a given shoe in the resale market. Additionally, PCA proved useful for reducing features and could help improve the model's training time efficiency on future larger datasets. Clustering was revealed to be difficult to implement given the data, leading to possible ensemble learning to be impractical.

# Model Validity and Limitations
The accuracy and completeness of StockX data might be questionable as we only have a sliver of the shoes that StockX sells. Getting more data might increase their accuracy. Additionally, we aren’t directly modeling shoe sales, but rather the resale of them. This puts into question the validity of our models as they may only be applicable to the resale market. While there is an implied correlation between the retail and secondary markets, it is hard to say how much they are truly correlated.

# Appendix

Data Sources:
Kaggle StockX Data: https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest/data Code Repository: https://github.com/Analytics-2024/Data_Analysis.git






