# Monitoring The Future: 12th Grader Opiate Use - Part 1
#### J M Maxwell - Data Science, Sr. Analyst - CTDS
## Analysis Introduction
#### About The Data


The [Monitoring The Future](https://monitoringthefuture.org) surveys are a series of surveys that have emerged as a vital tool in measuring the values, behaviors, and lifestyle orientations among American youth. Through a comprehensive series of surveys, these studies offer a unique window into the ever-changing landscape of 12th grade students.

Students are randomly assigned one of six questionnaires; each questionnaire comprises both a core set of questions common to all surveys and a set of questions tailored to the specific survey, collectively providing a rich dataset for exploration. There are approximately 1,400 variables across all of the questionnaires; while recognizing the vastness of the available data, our exploration will be primarily focused on a subset of variables deemed particularly relevant to our research objectives.

One of the critical aspects examined in the surveys pertains to the frequency of drug use among students, encompassing a wide array of illicit and recreational substances. In the context of this analysis, specific attention has been limited to surveys which observe instances where students engaged in the use of heroin or other opioid narcotics.

While the investigation of drug use patterns holds significance in understanding the landscape of contemporary American youth, the Monitoring The Future surveys provide a broader canvas for exploration. Our analysis encompasses an assortment of other captivating topics including, but not limited to, students' perspectives on religion, educational goals, family life dynamics, and work habits. By examining these multidimensional facets, we may gain a holistic understanding of the myriad factors influencing the lives, aspirations, and substance use risks of American youth.

#### Import Python Packages And Data


```python
!pip install matplotlib -q
!pip install scikit-learn -q
!pip install imblearn -q
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.utils import resample

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from IPython.display import Markdown

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


random_seed=2023

df_12_2019_1 = pd.read_csv('Grade12/ICPSR_37841/DS0001/37841-0001-Data.tsv', sep='\t')
df_12_2019_3 = pd.read_csv('Grade12/ICPSR_37841/DS0003/37841-0003-Data.tsv', sep='\t')
```

#### Data Cleaning And Feature Selection

Below is a key for mapping the surveys' coded variable names to a more interpretable set of variable labels.

We identified a number variables from the original dataset that needed to be removed because they were highly correlated. For instance, we observed among the survey recipients an unsurprising association between the number of alcoholic drinks consumed this year and the number of alcoholic drinks consumed in the last 30 days. We believe that the latter variable will serve as a more robust indicator for opiate use, so to streamline our analysis and enhance the veracity of our work, we made the decision to remove the former variable.

The initial dataset comprises an impressive collection of 16,000+ survey results, each representing an observation from a survey participant. However, it is important to note that a substantial portion of these observations were removed during the data cleaning process, as those observations were missing portions of their data.

Most of the data you see below is either recorded as a binary outcome (1/yes or 0/no), as an ordinal value(i.e., 0 = no drug use, 1 = some drug use, 2 = frequent drug use), or as a categorical feature. The categorical data is encoded and represented as binary data.


```python
variable_dict = {
 'RESPONDENT_AGE': 'Over18',
 'V13': 'SchoolRegion',
 'V49': 'NumberOfSiblings',
 'V2102': 'CigsSmoked/30Days',
 'V2106': 'AlcoholicDrinksHowManyTimes/30Days',
 'V2117': 'MarijuanaHowManyTimes/30Days',
 'V2118': 'LSDHowManyTimes/Life',
 'V2121': 'PsychedelicsHowManyTimes/Life',
 'V2124': 'CocaineHowManyTimes/Life',
 'V2127': 'AmphetaminesHowManyTimes/Life',
 'V2133': 'SedativesHowManyTimes/Life',
 'V2136': 'TranquilizersHowManyTimes/Life',
 'V2139': 'HerHowManyTimes/Life',
 'V2142': 'NarcHowManyTimes/Life',
 'V2150': 'Sex',
 'V2151': 'Race',
 'V2152': 'RaisedWhere',
 'V2153': 'MaritalStatus',
 'V2155': 'LivesWithFather',
 'V2156': 'LivesWithMother',
 'V2157': 'LivesWithSiblings',
 'V2163': 'FatherEduLvl',
 'V2164': 'MotherEduLvl',
 'V2165': 'MotherHadPaidJobWhileGrowingUp',
 'V2166': 'PoliticalPreference',
 'V2167': 'PoliticalBeliefs',
 'V2169': 'ReligiousServiceAttendenceWkly',
 'V2170': 'ReligionImportance',
 'V2172': 'HighSchoolProgram',
 'V2174': 'SelfRateIntelligence',
 'V2175': 'SchoolDaysMissedIllness/4Weeks',
 'V2176': 'SchoolDaysMissedSkipped/4Weeks',
 'V2177': 'SchoolDaysMissedOther/4Weeks',
 'V2178': 'SkippedClass/4Weeks',
 'V2179': 'AverageGradeHS',
 'V2180': 'LikelyToAttendVocationalSchl',
 'V2181': 'LikelyToServeInMilitary',
 'V2182': 'LikelyToGraduate2YrCollege',
 'V2183': 'LikelyToGraduate4YrCollege',
 'V2184': 'LikelyToAttendGraduateSchl',
 'V2185': 'WantToDoVocationalSchl',
 'V2186': 'WantToServeInMilitary',
 'V2187': 'WantToDo2YrCollege',
 'V2188': 'WantToDo4YrCollege',
 'V2189': 'WantToDoGradSchl',
 'V2190': 'WantToDoNo2ndEd',
 'V2191': 'HrsWorkedPerWeek',
 'V2193': 'MoneyFromOtherSource',
 'V2194': 'EveningsOutPerWeek',
 'V2195': 'DatesHowOften',
 'V2196': 'MilesDrivenPerWeek',
 'V2197': 'DrivingTickets',
 'V2201': 'CarAccidentsLast12Mo',
 'V2459': 'CrackHowManyTimes/Life',
}
```

The folowing steps were taken to clean the data:

1) We filtered the existing data by removing all variables missing more than 30% of their resepective values. We then remove all observations missing any survey responses.

2) We combined the survey questions asking students how many times they have used heroin and how many times they used opioid narcotics in their life into a single variable marking whether they have ever used heroin or an opiate.

3) We renamed all of our variables using the coded data dictionary above.

4) We factored the categorical data so it can be represented numerically.

5) We normalized the data so the ordinal and numerical outcomes do not have an oversized effect on our models.

After these steps are complete we are left with only 5377 survey observations, approximately a third of our initial data.


```python
# Filter data down to just the variable dictionary, removing correlated features in the process
variables = list(variable_dict.keys())
df = pd.concat([df_12_2019_1[variables],df_12_2019_3[variables]], ignore_index=True)

# Remove missing data
missing_criteria = (df == -9).sum() < 0.3*len(df.index)
df = df[missing_criteria.index[missing_criteria]]

df_counts = df.apply(pd.Series.value_counts, axis=1)
missing_data = df_counts.iloc[:, 0]
missing_data = missing_data.fillna(0)
minimal_missing = missing_data.index[missing_data < 1]
df = df[df.index.isin(minimal_missing)]

# Combine Opiate Use data
df['OpiateUse'] = ((df['V2142'] != 1) + (df['V2139'] != 1)).astype(int)
df = df.drop(['V2142', 'V2139'], axis=1)

# Rename columns using data dictionary
df.rename(columns=variable_dict, inplace=True)

# Factor categorical data
dummy_cols = ['SchoolRegion', 'Race', 'RaisedWhere', 'MaritalStatus', 'PoliticalPreference', 'PoliticalBeliefs', 'HighSchoolProgram']
dummies = pd.get_dummies(df[dummy_cols], columns=dummy_cols, drop_first=True)
df = pd.concat([df, dummies], axis=1)
df = df.drop(dummy_cols, axis=1)

# Normalize data
df.replace({False: 0, True: 1}, inplace=True)
df = (df-df.min())/(df.max()-df.min())

df = df.reset_index(drop=True)
```

## Model Selection

We began our model selection stage by training five different machine learning models and comparing their performance. We then selected the two best performing models. The five initial models we trained were a simple logistic regression model, a lasso logistic regression model, a ridge logistic regression model, a support vector machine (SVM) classifier, and a random forest classifier.


#### Model Overview

- [Logistic Regression Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): A widely used technique for exploring relationships between predictor variables and the probability of a set of binary outcomes occuring.

- [Lasso Logistic Regression Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): A variation of the logistic regression model that incorporates a component encouraging the model to minimize the number of predictor variables, helping us identify which predictor variables are most (and least) influential.

- [Ridge Logistic Regression Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): Another variation of the logistic regression model that promotes a balanced selection of predictor variables, allowing us to more clearly compare the influence of different predictor variables on opiate use.

- [Support Vector Machine (SVM) Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): a classification model specificly suited to uncover complex boundaries to distinguish between classes of opiate users and non-users.

- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): Utilizes the collective descision making power of an ensemble of smaller, simpler models (decison trees) to produce more robust insights on the factors associated with opiate use in students.

#### [Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

In order to make a precise comparison between these five models, we need to utilize cross-validation. Cross-validation involves executing multiple iterations of training and testing a model on various random, resampled selections of the data. This method is primarily used to estimate how a predictive model will perform in practice. In our use case, we compared the performance of our five models during cross-validation to determine which models perform best.

#### Resolving Class Imbalances

As expected, a limited number of the students surveyed had ever used opiates. Only 295 of the 5377 (approx. 5.5%) students had ever used opiates or heroin recreationally. This imbalance in classes made it very difficult to train a model to predict occurances of the minority class (opiate use) accurately. To resolve this problem, we used a technique called [Synthetic Minority Oversampling Technique (SMOTE)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html). SMOTE synthesizes new examples from the minority class by creating synthetic data points which only slightly differ from the original data points of the minority class.


#### SMOTE Cross Validation Pipeline

Below you will find the function for a pipeline for performing model training and testing using 10-fold cross validation and SMOTE and for recording each model's cross validated performace.


```python
def pipeline_cross_validation(data, k, pipeline_steps):

    folds = np.array_split(data, k)
    accuracySum = 0
    recallSum = 0
    precisionSum = 0

    for i in range(k):
        train = folds.copy()
        test = folds[i]
        del train[i]
        train = pd.concat(train, sort=False)

        y_train = train.OpiateUse.astype(int)
        X_train = train.drop('OpiateUse', axis=1)

        y_test = test.OpiateUse.astype(int)
        X_test = test.drop('OpiateUse', axis=1)

        pipeline = Pipeline(pipeline_steps).fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracySum += accuracy_score(y_test, y_pred)
        recallSum += recall_score(y_test, y_pred)
        precisionSum += precision_score(y_test, y_pred)


    return [accuracySum/k, recallSum/k, precisionSum/k]
```

#### Model Cross Validation


```python
k = 10
df2 = df.iloc[np.random.permutation(len(df))]
classification_scores = pd.DataFrame({'Metric': ['Accuracy', 'Recall', 'Precision']})
smt = SMOTE(random_state=random_seed)

model = LogisticRegression(solver='liblinear')
steps = [('smt', smt), ('model', model)]
classification_scores['Logistic Regression'] = pipeline_cross_validation(df2, k, steps)

model = LogisticRegression(solver='liblinear', penalty='l1')
steps = [('smt', smt), ('model', model)]
classification_scores['Lasso Logistic Regression'] = pipeline_cross_validation(df2, k, steps)

model = LogisticRegression(solver='liblinear', penalty='l2')
steps = [('smt', smt), ('model', model)]
classification_scores['Ridge Logistic Regression'] = pipeline_cross_validation(df2, k, steps)

model = SVC(C=1.5, kernel='rbf')
steps = [('smt', smt), ('model', model)]
classification_scores['Support Vector Machine'] = pipeline_cross_validation(df2, k, steps)

model = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True)
steps = [('smt', smt), ('model', model)]
classification_scores['Random Forest Classifier'] = pipeline_cross_validation(df2, k, steps)
```

#### Model Classification Scores

Below we find the model accuracy, recall, and precision scores for all five models. All three metrics are scored between 0.0 and 1.0, with 1.0 being a perfect score. In our current application, model accuracy is a measurement of how well the model correctly labels whether a student is likely to either have used opiates/heroin or to not have used opiates/heroin. Recall measures how well the model performs at detecting all of the subjects who have used opiates/heroin. Precision measures how well the model performs at accurately specifying which subjects have used opiates/heroin from those subjects who have not used opiates/heroin.

While we resolved the class imbalance when training our models, the class imbalance can still effect our scores, particularly model accuracy during testing. With that in mind, we primarily relied on the recall and precision scores when comparing the performance of the models and selecting our final two models.



```python
Markdown(classification_scores.head().to_markdown())
```




|    | Metric    |   Logistic Regression |   Lasso Logistic Regression |   Ridge Logistic Regression |   Support Vector Machine |   Random Forest Classifier |
|---:|:----------|----------------------:|----------------------------:|----------------------------:|-------------------------:|---------------------------:|
|  0 | Accuracy  |              0.863121 |                    0.866843 |                    0.863121 |                 0.961129 |                   0.967452 |
|  1 | Recall    |              0.704792 |                    0.703083 |                    0.704792 |                 0.524306 |                   0.515776 |
|  2 | Precision |              0.244013 |                    0.250618 |                    0.244013 |                 0.699461 |                   0.80553  |



#### Model Selection

We chose one model with the highest recall score and one with the highest precision score. The random forest classifier clearly had the best precison score. The three logistic regression models all had very similar recall scores, however, we chose the lasso logistic regression model from those three models. The lasso logistic regression model will typically be more interpretable and simpler by reducing the number of variables.

We proceeded with retraining the random forest classifier and lasso logistic regression models so we could examine what each model believes to be the most influential variables on whether a 12th grader has ever used opiates/heroin.

## Model Analysis


```python
X = df.drop('OpiateUse', axis=1)
y = df.OpiateUse.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
classification_scores = pd.DataFrame({'Metric': ['Accuracy', 'Recall', 'Precision']})
```


```python
model = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True)
steps = [('smt', smt), ('model', model)]
pipeline = Pipeline(steps).fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

feature_importance = pd.DataFrame({'RF Feature Importance': pipeline['model'].feature_importances_}, index=df.columns.drop('OpiateUse'))
feature_importance['Normalized RF Feature Importance'] = (pipeline['model'].feature_importances_ - min(pipeline['model'].feature_importances_)) / (max(pipeline['model'].feature_importances_) - min(pipeline['model'].feature_importances_))
classification_scores['Random Forest Classifier'] = [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred),  precision_score(y_test, y_pred)]
```


```python
model = LogisticRegression(solver='liblinear', penalty='l1')
steps = [('smt', smt), ('model', model)]
pipeline = Pipeline(steps).fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

feature_importance['Lasso Model Coefficients'] = pipeline['model'].coef_[0]
feature_importance['Lasso Model Coefficients AbsVal'] = abs(pipeline['model'].coef_[0])
classification_scores['Lasso Logistic Regression'] = [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred),  precision_score(y_test, y_pred)]
```

Again we see similar recall and precision scores as before.


```python
Markdown(classification_scores.to_markdown())
```




|    | Metric    |   Random Forest Classifier |   Lasso Logistic Regression |
|---:|:----------|---------------------------:|----------------------------:|
|  0 | Accuracy  |                   0.958178 |                    0.859665 |
|  1 | Recall    |                   0.476923 |                    0.753846 |
|  2 | Precision |                   0.738095 |                    0.266304 |



#### Lasso Logistic Regression Influential Variables

We first looked at the most influential variables in the lasso logistic regression model. Below are a rank ordered list of the 15 most influential variables. Because the lasso logistic regression model's coefficients are related to a high recall score, they let us know which of the variables are the most influential on the model's ability to identify all of the students who have used opiates/heroin. This particular model was able to identify between 70% and 75% of all the surveyed students who had used opiates/heroin.

In the data frame below, 'Lasso Model Coefficients' is the coefficient in the model and tells us the relative influence of each of the 10 variables with the greatest positive influence towards identifying students who used opiates. For instance, the frequency of with which students use tranquilizers or LSD (the two highest ranked variables) are highly influential in identifying students who have ever used opiates. Furthermore, 8 of top 10 variables that are highly related to student's opiate use involve the student's illicit use of other drugs or alcohol.

When comparing the influence of each variable in our model, it is important to note the relative scale of the model coefficients. For instance, each use of amphetamines or cocaine in a students lifetime is approximately 3 times more influential to our model's prediction of opiate use than each use of marijuana or sedatives/barbituates.


```python
Markdown(feature_importance.sort_values(by=['Lasso Model Coefficients'], axis=0, ascending=False)[['Lasso Model Coefficients']].head(15).to_markdown())
```




|                                    |   Lasso Model Coefficients |
|:-----------------------------------|---------------------------:|
| TranquilizersHowManyTimes/Life     |                   7.3497   |
| LSDHowManyTimes/Life               |                   6.9445   |
| CocaineHowManyTimes/Life           |                   4.26338  |
| AmphetaminesHowManyTimes/Life      |                   4.14465  |
| CigsSmoked/30Days                  |                   2.13381  |
| AlcoholicDrinksHowManyTimes/30Days |                   1.69848  |
| MarijuanaHowManyTimes/30Days       |                   1.54761  |
| SedativesHowManyTimes/Life         |                   1.22001  |
| LivesWithFather                    |                   0.978733 |
| MotherHadPaidJobWhileGrowingUp     |                   0.821343 |
| WantToServeInMilitary              |                   0.697732 |
| AverageGradeHS                     |                   0.606838 |
| LikelyToAttendVocationalSchl       |                   0.595602 |
| LikelyToAttendGraduateSchl         |                   0.466966 |
| MotherEduLvl                       |                   0.348325 |



#### Lasso Logistic Regression Least Influential Variables

According to our lasso logistic regression model, these are the 10 least influential variables on whether a survey subject was likely or unlikely to use opiates/heroin. These attributes give little to no indication into whether a student will have illicitly used opiates or heroin.  These variables are listed in order of least influence.
The relevant encodings on this list are:

- 'PoliticalBeliefs_6' - Radical (ranked 1-6 from very conservative to radically liberal, with option for no belief)
- 'PoliticalBeliefs_3' - Moderate (ranked 1-6 from very conservative to radically liberal, with option for no belief)
- 'Race_3' - Hispanic (subjects self reported race as Black, White, or Hispanic)

From our model, it appears that some student's political beliefs, the number of times the student has skipped class in the last four weeks, a lack of desire to pursure secondary education, and the number of psychedelics they have consumed in their lifetime all provide little to no information as to whether a student would or would not use opiates/heroin.


```python
Markdown(feature_importance.sort_values(by=['Lasso Model Coefficients AbsVal'], axis=0, ascending=True)[['Lasso Model Coefficients AbsVal']].head(10).to_markdown())
```




|                                |   Lasso Model Coefficients AbsVal |
|:-------------------------------|----------------------------------:|
| PsychedelicsHowManyTimes/Life  |                         0         |
| SkippedClass/4Weeks            |                         0         |
| WantToDoNo2ndEd                |                         0         |
| PoliticalBeliefs_3             |                         0         |
| PoliticalBeliefs_6             |                         0         |
| Race_3                         |                         0.0410165 |
| DatesHowOften                  |                         0.0453305 |
| LikelyToServeInMilitary        |                         0.0460795 |
| NumberOfSiblings               |                         0.0713443 |
| ReligiousServiceAttendenceWkly |                         0.0808742 |



#### Random Forest Influential Variables

Now, let's analyze the key attributes of the random forest classification model. Presented below is a ranked list of the top 15 most important variables to our random forest model. These variables play a crucial role in achieving a high precision score and provide insights into the most influential factors affecting the model's capability to accurately distinguish and identify students who have used opiates/heroin (it is highly unlikely that the model will indicate that student who has never used opiates/heroin as having used opiates/heroin at least once in their life).

In the data frame below, 'RF Feature Importance' is the Gini Feature Importance of the random forest model and 'Normalized RF Feature Importance' is the the same set of values normalized to a range between 0 and 1. As with our lasso logistic regression model, a variety of other drug and alcohol use is highly influential on whether a subject is likely to have or have not used opiates/heroin. Other important features include how often the subject goes out on dates, the self reported importance of religion and attendence to religious services, the number school days they have missed in the last four weeks, how many hours they work each week, and whether the subject's mother had a paid job growing up. Unlike the coefficients of our ridge logistic regression model above, these scores do not tell us whether a particular variable has a strong postive or negative influence on predicting whether a student has or has not used opiates/heroin.


```python
Markdown(feature_importance.sort_values(by=['RF Feature Importance'],
                               axis=0, ascending=False)[['RF Feature Importance', 'Normalized RF Feature Importance']].head(15).to_markdown())
```




|                                    |   RF Feature Importance |   Normalized RF Feature Importance |
|:-----------------------------------|------------------------:|-----------------------------------:|
| TranquilizersHowManyTimes/Life     |               0.104334  |                           1        |
| AmphetaminesHowManyTimes/Life      |               0.0943735 |                           0.90435  |
| AlcoholicDrinksHowManyTimes/30Days |               0.0748839 |                           0.717192 |
| MarijuanaHowManyTimes/30Days       |               0.0633785 |                           0.606706 |
| LSDHowManyTimes/Life               |               0.0550314 |                           0.526549 |
| SedativesHowManyTimes/Life         |               0.0350997 |                           0.335146 |
| CocaineHowManyTimes/Life           |               0.0254969 |                           0.242931 |
| CigsSmoked/30Days                  |               0.0233875 |                           0.222674 |
| ReligionImportance                 |               0.0198766 |                           0.188959 |
| SchoolDaysMissedSkipped/4Weeks     |               0.0195713 |                           0.186027 |
| PsychedelicsHowManyTimes/Life      |               0.0194071 |                           0.18445  |
| HrsWorkedPerWeek                   |               0.0192343 |                           0.182791 |
| FatherEduLvl                       |               0.0183928 |                           0.174711 |
| MotherHadPaidJobWhileGrowingUp     |               0.0183627 |                           0.174421 |
| LikelyToAttendVocationalSchl       |               0.018341  |                           0.174213 |



## Conclusions

By employing SMOTE and cross-validation techniques, we successfully identified the random forest classifier and lasso logistic regression model as two effective models for accurately predicting the likelihood of survey subjects having used opiates or heroin during their 12th grade year. Using these models, we evaluated the influential variables that impacted our predictions.

Many of the identified important variables aligned with expectations. It was not surprising to find that students who engaged in various illicit drug use and underage drinking were at higher risk of using opiates and heroin. Additionally, we discovered that a number of factors such as the student's desire to pursue secondary education or the freuency of skipping class had a minimal impact on the lasso logistic regression model's outcomes. This suggests that these variables may not be significant predictors of opiate/heroin usage according to the model.



## Future Work
Future work is planned in order to incorporate the ICPSR *Monitoring the Future* data for years outside of 2019. We would like to analyze the survey results from years prior to the COVID-19 pandemic in comparison to survey results during the pandemic in order to examine how 12th grader opiate use and the factors influencing their opiate use have changed over time.
