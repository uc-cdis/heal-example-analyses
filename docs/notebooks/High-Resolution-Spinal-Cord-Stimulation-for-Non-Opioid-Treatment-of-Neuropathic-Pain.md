# Analysis Of A Non-Opioid Treatment For Neuropathic Pain (HDP00384)

#### By J M Maxwell - Data Science, Sr. Analyst - CTDS

In this notebook, we'll be analyzing data from the study [*High-Resolution, Spinal Cord Stimulation for Non-Opioid Treatment of Neuropathic Pain*](https://data.mendeley.com/datasets/rmj2kngzbp/1) by Bryan McLaughlin. The study was conducted in two phases, and this analysis utilizes the pre and post operation results from five indices for the subject's self reported pain, depression, or pain inducing disability.

This work was strictly done to demonstrate the advantages of the HEAL Platform's Workspace feature and the ability to utilize data that is joined under the HEAL data mesh. All the following work was completed by J M. Maxwell and members of the HEAL Platform team, but was based on the data provided in [*High-Resolution, Spinal Cord Stimulation for Non-Opioid Treatment of Neuropathic Pain*](https://data.mendeley.com/datasets/rmj2kngzbp/1) and influenced by the work of McLaughlin et al. in [*Correlating Evoked Electromyography and Anatomic Factors During Spinal Cord Stimulation Implantation With Short-Term Outcomes*](https://pubmed.ncbi.nlm.nih.gov/39320285/).

The work here does not represent the official opinions, recommendations, or conclusions of Bryan McLaughlin and this work does not represent policy or medical recommendations on behalf of the NIH HEAL Initiative, The Center For Translational Data Science, or The University of Chicago.

    McLaughlin, Bryan (2024), “High-Resolution, Spinal Cord Stimulation for Non-Opioid Treatment of Neuropathic Pain (U44NS115111)”, Mendeley Data, V1, doi: 10.17632/rmj2kngzbp.1

## Access Data

To access the data from this study make sure you are logged in to the InCommon login option and then:
1) Go to the HEAL Discovery page to select the study
2) Select the 'Open In Workspace' option and choose the (Tutorials) Example Analysis Jupyter Lab Notebooks workspace option
3) Use the exported study manifest to download the study.

Otherwise you may run the following gen3-sdk command to download the relevant CSV file from the study.


```python
!gen3 drs-pull object dg.H34L/b8b871b8-aadd-4017-9f67-184b17ab3580
!gen3 drs-pull object dg.H34L/ffde8647-1ec2-4459-8409-bd41c0736c86
```

    {"succeeded": ["dg.H34L/b8b871b8-aadd-4017-9f67-184b17ab3580"], "failed": []}
    {"succeeded": ["dg.H34L/ffde8647-1ec2-4459-8409-bd41c0736c86"], "failed": []}


## About the Study

The study, *High-Resolution, Spinal Cord Stimulation for Non-Opioid Treatment of Neuropathic Pain*, investigated the outcomes of elliciting EMGs (electromyography) in subject's regions of pain during surgery. Data for evaluating the treatment effect were collected from 21 patients in two phases. Subject outcomes were measured using: the Numerical Rating Scale, McGill Pain Questionnaire, Beck Depression Inventory, Oswestry Disability Index, and Pain Catastrophizing Score, and were recorded preoperatively and at three months following the procedure.

### Load Packages and Data


```python
!pip install matplotlib -q

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
import os

pd.set_option('max_colwidth', 800)
from IPython.display import Markdown, Image, display

os.makedirs('img/Non-Opioid_Treatment_Analysis')
```

### Read In Data and Clean Data

After reading in the data, we perform standard data cleaning steps to improve the usability of the data, remove extraneous data features, and engineer new features measuring the percent change in treatment effects from before and after the each sample subject's operation.


```python
participants_df = pd.read_excel('Microleads - Participant level data.xlsx')
pain_scores_df = pd.read_excel('Pain-MRI scores.xlsx')
pain_scores_df = pain_scores_df.iloc[:21, :]
df = pd.merge(left=participants_df, right=pain_scores_df, how='left', on=['Phase', 'Patient'])
df[list(df.select_dtypes(include='float64'))] = df[list(df.select_dtypes(include='float64'))].astype('float32')
df.replace('-', np.nan, inplace=True)
df.rename(columns={'NRS Pre-op': 'NRS Pre-Op', 'NRS Post-op': 'NRS Post-Op', 'MPQ pre-op': 'MPQ Pre-Op', 'MPQ post-op': 'MPQ Post-Op', 'Gender': 'Sex'}, inplace=True)
df.drop(['Patient', 'Race', 'Ethnicity', 'Age Unit'], axis=1, inplace=True)

delta_cols = ['NRS', 'MPQ', 'ODI', 'PCS', 'BDI']
for col in delta_cols:
    df[f'{col}_pct_change'] = np.round(100*((df[f'{col} Post-Op'] - df[f'{col} Pre-Op'])/df[f'{col} Post-Op']), 2)
df.replace(np.float64('-inf'), np.nan, inplace=True)

Markdown(df.to_markdown())
```




|    |   Phase | Sex   |   Age |   NRS Pre-Op |   NRS Post-Op |   MPQ Pre-Op |   MPQ Post-Op |   ODI Pre-Op |   ODI Post-Op |   PCS Pre-Op |   PCS Post-Op |   BDI Pre-Op |   BDI Post-Op |   AP column Diameter (mm) |   Interpedicular distance (mm) |   Dorsal CSF Thickness (mm) |   NRS_pct_change |   MPQ_pct_change |   ODI_pct_change |   PCS_pct_change |   BDI_pct_change |
|---:|--------:|:------|------:|-------------:|--------------:|-------------:|--------------:|-------------:|--------------:|-------------:|--------------:|-------------:|--------------:|--------------------------:|-------------------------------:|----------------------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|  0 |       1 | F     |    54 |          9   |             4 |            6 |             6 |         68   |            44 |           14 |             7 |           10 |             5 |                      12.8 |                           16.5 |                         3   |          -125    |             0    |           -54.55 |          -100    |          -100    |
|  1 |       1 | M     |    60 |          9   |             5 |            8 |             7 |         62   |            36 |           34 |            16 |           15 |            11 |                      15.8 |                           20   |                         4   |           -80    |           -14.29 |           -72.22 |          -112.5  |           -36.36 |
|  2 |       1 | M     |    46 |          7   |             3 |            9 |             8 |         54   |            30 |           17 |             4 |           17 |             7 |                      11.3 |                           14.8 |                         2.5 |          -133.33 |           -12.5  |           -80    |          -325    |          -142.86 |
|  3 |       1 | M     |    29 |          8   |             5 |            9 |             5 |         52   |            54 |           20 |             7 |            5 |             8 |                      15.2 |                           19.5 |                         5   |           -60    |           -80    |             3.7  |          -185.71 |            37.5  |
|  4 |       1 | F     |    43 |          8   |             3 |            6 |             5 |         78   |           nan |           42 |            38 |           46 |            23 |                      14.8 |                           19.6 |                         5   |          -166.67 |           -20    |           nan    |           -10.53 |          -100    |
|  5 |       1 | F     |    40 |          2   |             0 |            5 |           nan |         12   |           nan |            2 |           nan |            4 |           nan |                      17   |                           21.5 |                         4.8 |           nan    |           nan    |           nan    |           nan    |           nan    |
|  6 |       1 | F     |    52 |          7   |             8 |            4 |             5 |         60   |            62 |            2 |            26 |            7 |            16 |                      17.6 |                           18.8 |                         3.9 |            12.5  |            20    |             3.23 |            92.31 |            56.25 |
|  7 |       1 | M     |    67 |          6   |             4 |            1 |           nan |         62   |           nan |           24 |           nan |           19 |           nan |                      19   |                           19.3 |                         5.3 |           -50    |           nan    |           nan    |           nan    |           nan    |
|  8 |       1 | F     |    57 |          7   |             7 |            1 |            14 |         73   |            74 |           14 |             6 |           16 |             5 |                      18.1 |                           20   |                         5.4 |             0    |            92.86 |             1.35 |          -133.33 |          -220    |
|  9 |       1 | F     |    36 |         10   |             8 |            9 |            13 |         76   |            76 |           39 |            23 |           34 |            40 |                      17   |                           21.8 |                         5.7 |           -25    |            30.77 |             0    |           -69.57 |            15    |
| 10 |       1 | M     |    71 |          6   |           nan |            5 |           nan |         60   |           nan |           39 |           nan |           23 |           nan |                      13.9 |                           21.3 |                         3.9 |           nan    |           nan    |           nan    |           nan    |           nan    |
| 11 |       1 | F     |    78 |          5   |             5 |            4 |            11 |         37.8 |            31 |           30 |            35 |           21 |            14 |                      18   |                           21.5 |                         4.7 |             0    |            63.64 |           -21.94 |            14.29 |           -50    |
| 12 |       2 | F     |    57 |          8   |             3 |           11 |             5 |         64   |            36 |           27 |             7 |           19 |            12 |                     nan   |                          nan   |                       nan   |          -166.67 |          -120    |           -77.78 |          -285.71 |           -58.33 |
| 13 |       2 | F     |    60 |          8   |             3 |            4 |             2 |         56   |            38 |           42 |            22 |           27 |            15 |                      15   |                           17.6 |                         4.3 |          -166.67 |          -100    |           -47.37 |           -90.91 |           -80    |
| 14 |       2 | F     |    67 |          8   |             0 |            5 |             0 |         64   |             0 |           24 |             0 |           13 |             0 |                      14.1 |                           17.3 |                         3.1 |           nan    |           nan    |           nan    |           nan    |           nan    |
| 15 |       2 | F     |    51 |          8   |             6 |            4 |            15 |         60   |            50 |            8 |            42 |            0 |             7 |                      14.4 |                           17.5 |                         4.8 |           -33.33 |            73.33 |           -20    |            80.95 |           100    |
| 16 |       2 | F     |    78 |          8   |             8 |            8 |             8 |         38   |            42 |           26 |            29 |            5 |             7 |                      15.2 |                           19.3 |                         4.9 |             0    |             0    |             9.52 |            10.34 |            28.57 |
| 17 |       2 | M     |    75 |          8   |             2 |           11 |             4 |         49   |            40 |           11 |             8 |            5 |             2 |                      12.2 |                           18   |                         4.7 |          -300    |          -175    |           -22.5  |           -37.5  |          -150    |
| 18 |       2 | F     |    34 |          8   |             5 |           21 |            10 |         62   |            50 |           38 |            41 |           33 |            32 |                      14.4 |                           17.2 |                         3.6 |           -60    |          -110    |           -24    |             7.32 |            -3.12 |
| 19 |       2 | M     |    71 |          6.5 |             2 |           22 |             0 |         30   |             4 |           19 |             1 |            9 |             2 |                      12.3 |                           21.4 |                         5.6 |          -225    |           nan    |          -650    |         -1800    |          -350    |
| 20 |       2 | F     |    49 |          7   |             3 |           13 |             4 |         42   |            21 |           29 |            24 |            7 |            14 |                      14.8 |                           18.7 |                         6.2 |          -133.33 |          -225    |          -100    |           -20.83 |            50    |



## Summary Statistics

First, let us look at the summary statistics for our measured treatment effects and our engineer treatment effects.


```python
characteristics = ['Sex M/F', 'Age', 'Anterior-posterior diameter (mean ± SD in mm)',
                   'Interpedicular distance (mean ± SD in mm)', 'Dorsal CSF thickness (mean ± SD in mm)',
                   'Numerical Rating Scale (mean ± SD)', 'McGill Pain Questionnaire (mean ± SD)',
                   'Oswestry Disability Index (mean ± SD)', 'Pain Catastrophizing Scale (mean ± SD)', 'Beck Depression Index (mean ± SD']

cols = ['Age', 'AP column Diameter (mm)', 'Interpedicular distance (mm)', 'Dorsal CSF Thickness (mm)',
        'NRS Pre-Op', 'MPQ Pre-Op',  'ODI Pre-Op',  'PCS Pre-Op',  'BDI Pre-Op', ]

x = df.Sex.value_counts().values
values = [f'{int(x[1])} / {int(x[0])}']
for col in cols:
    values.append(f'{np.round(float(df[col].mean()), decimals=1)} ' + u"\u00B1" + f' {np.round(float(df[col].std()), 1)}')

table_df1 = pd.DataFrame({'Patient Information and Clinical Characteristics': characteristics, 'Values': values})
Markdown(table_df1.to_markdown())
```




|    | Patient Information and Clinical Characteristics   | Values      |
|---:|:---------------------------------------------------|:------------|
|  0 | Sex M/F                                            | 7 / 14      |
|  1 | Age                                                | 56.0 ± 14.6 |
|  2 | Anterior-posterior diameter (mean ± SD in mm)      | 15.1 ± 2.1  |
|  3 | Interpedicular distance (mean ± SD in mm)          | 19.1 ± 1.9  |
|  4 | Dorsal CSF thickness (mean ± SD in mm)             | 4.5 ± 1.0   |
|  5 | Numerical Rating Scale (mean ± SD)                 | 7.3 ± 1.7   |
|  6 | McGill Pain Questionnaire (mean ± SD)              | 7.9 ± 5.5   |
|  7 | Oswestry Disability Index (mean ± SD)              | 55.2 ± 16.0 |
|  8 | Pain Catastrophizing Scale (mean ± SD)             | 23.9 ± 12.6 |
|  9 | Beck Depression Index (mean ± SD                   | 16.0 ± 11.7 |




```python
characteristics = ['Numerical Rating Scale Percent Change (mean ± SD)', 'McGill Pain Questionnaire Percent Change (mean ± SD)',
                   'Oswestry Disability Index Percent Change (mean ± SD)', 'Pain Catastrophizing Scale Percent Change (mean ± SD)',
                   'Beck Depression Index Percent Change (mean ± SD']

cols = ['NRS_pct_change', 'MPQ_pct_change', 'ODI_pct_change', 'PCS_pct_change', 'BDI_pct_change']
values = []
for col in cols:
    values.append(f'{np.round(float(df[col].mean()), decimals=1)} ' + u"\u00B1" + f' {np.round(float(df[col].std()), 1)}')

table_df2 = pd.DataFrame({'Patient Information and Clinical Characteristics': characteristics, 'Values': values})
Markdown(table_df2.to_markdown())
```




|    | Patient Information and Clinical Characteristics      | Values         |
|---:|:------------------------------------------------------|:---------------|
|  0 | Numerical Rating Scale Percent Change (mean ± SD)     | -95.1 ± 87.5   |
|  1 | McGill Pain Questionnaire Percent Change (mean ± SD)  | -36.0 ± 90.8   |
|  2 | Oswestry Disability Index Percent Change (mean ± SD)  | -72.0 ± 158.0  |
|  3 | Pain Catastrophizing Scale Percent Change (mean ± SD) | -174.5 ± 434.1 |
|  4 | Beck Depression Index Percent Change (mean ± SD       | -59.0 ± 113.6  |



## Statistical Testing

We're interested in comparing the treatment effects pre and post subject's operations. To begin with, we can use the Shapiro-Wilk Test to determine if the differences between each set of distributions are potentially normally distributed.
(h1)=
The null hypothesis is that the difference between the distributions of each treatment effect pre and post operation are not normally distributed. If we reject the null hypothesis, then we will proceed assuming the difference in distributions is not normally distributed. If we fail to reject the null hypothesis, then we will assume the distributions could be normally distributed.


```python
fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # Adjust the figure size as needed
axes = axes.flatten()

delta_cols = ['NRS', 'MPQ', 'ODI', 'PCS', 'BDI']
for i in range(0, len(delta_cols)):

    data = (df[f'{delta_cols[i]} Post-Op'] - df[f'{delta_cols[i]} Pre-Op']).dropna()
    stat, p = stats.shapiro(data)

    print(f'Shapiro-Wilk Test: {delta_cols[i]} \nStatistic: {np.round(stat, 3)} \nP-value: {np.round(p, 3)}')
    if p >  0.05:
        print(f'Sample {delta_cols[i]} could be normally distributed (fail to reject null hypothesis)')
    else:
        print(f'Sample {delta_cols[i]} does not look normally distributed (reject null hypothesis)')
    print("********************** \n")

    stats.probplot(data, dist="norm", plot=axes[i])
    axes[i].set_title(f"Normal Q-Q Plot {delta_cols[i]}")

plt.tight_layout()
axes[3].set_position([0.24,0.125,0.228,0.343])
axes[4].set_position([0.55,0.125,0.228,0.343])
axes[5].set_visible(False)
plt.close(fig)

fig.savefig('img/Non-Opioid_Treatment_Analysis/figure1.png')
Image(filename="img/Non-Opioid_Treatment_Analysis/figure1.png")
```

    Shapiro-Wilk Test: NRS
    Statistic: 0.958
    P-value: 0.513
    Sample NRS could be normally distributed (fail to reject null hypothesis)
    **********************

    Shapiro-Wilk Test: MPQ
    Statistic: 0.957
    P-value: 0.55
    Sample MPQ could be normally distributed (fail to reject null hypothesis)
    **********************

    Shapiro-Wilk Test: ODI
    Statistic: 0.862
    P-value: 0.016
    Sample ODI does not look normally distributed (reject null hypothesis)
    **********************

    Shapiro-Wilk Test: PCS
    Statistic: 0.881
    P-value: 0.027
    Sample PCS does not look normally distributed (reject null hypothesis)
    **********************

    Shapiro-Wilk Test: BDI
    Statistic: 0.96
    P-value: 0.607
    Sample BDI could be normally distributed (fail to reject null hypothesis)
    **********************







![png](output_12_1.png)





For three of the subjects' measured treatment effects, Numerical Rating Scale (NRS), McGill Pain Questionnairre (MPQ), and Beck Depression Index (BDI), we fail to reject the [null hypothesis](#h1) and so the differences between effects pre and post operation could be normally distributed. For these three we will test if the underlying distributions of the treatment effects pre and post operation have the same underlying distribution using the Student's t-test.

For the other two measured treatment effects, Oswestry Disability Index (ODI) and Pain Catastrophizing Scale (PCS), we reject the null hypthesis and assume the difference between treatment effects pre and post operation are not normally distributed. For these two treatment effects we cannot use the Student's t-test, and will instead use the non-parametric Mann Whitney U Test to test whether the the pre and post operation treatment effect samples hace the same underlying distribution. Because the Mann Whitney U Test is a non-parametric statistical test that does not assume normality, we will also use this test on the other three treatment effects.

### Student's T-Test

The Two-Sample, Paired Student's T-Test is used to test whether two related, sample distributions have a statistically significant difference.

(h2)=
Our null hypothesis is that the averages (or expected) values of the two samples, pre and post treatment effects, are the same.

If the null hypothesis is rejected, then the pre and post operation treatment effect samples do not have identical average (expected) values and the pre and post operation samples could be different, or in other words, the operation may have influenced a change in the measured treatment effect.

If we fail to reject the null hypothesis, then the pre and post operation treatment effect samples could have identical average (expected) values and the operation may not have influenced a change in the measured treatment effect.


```python
delta_cols = ['NRS', 'MPQ', 'BDI']
for col in delta_cols:
    resultTtest = stats.ttest_rel(a=df[f'{col} Pre-Op'], b=df[f'{col} Post-Op'], nan_policy='omit')

    print(f"Student's t Test: {col} \nStatistic: {np.round(resultTtest.statistic, 3)} \nP-value: {np.round(resultTtest.pvalue, 3)}")
    if resultTtest.pvalue >  0.05:
        print(f'Pre and Post Op {col} samples could have identical average (expected) values. (fail to reject null hypothesis)')
    else:
        print(f'Pre and Post Op {col} samples do not have identical average (expected) values. (reject null hypothesis)')
    print("********************** \n")
```

    Student's t Test: NRS
    Statistic: 6.139
    P-value: 0.0
    Pre and Post Op NRS samples do not have identical average (expected) values. (reject null hypothesis)
    **********************

    Student's t Test: MPQ
    Statistic: 0.961
    P-value: 0.35
    Pre and Post Op MPQ samples could have identical average (expected) values. (fail to reject null hypothesis)
    **********************

    Student's t Test: BDI
    Statistic: 1.916
    P-value: 0.072
    Pre and Post Op BDI samples could have identical average (expected) values. (fail to reject null hypothesis)
    **********************



We only reject the [null hypothesis](#h2) for the first treatment effect, Numerical Rating Scale (NRS), and fail to reject the [null hypothesis](#h2) for the other two treatment effects McGill Pain Questionnairre (MPQ) and Beck Depression Index (BDI).

### Wilcoxon Signed-Rank Test

The Wilcoxon Signed-Rank Test is a non-parametric version of the Student's T-Test used to test if there is a difference between two paired distributions.
(h3)=
It tests the null hypothesis that differences between the paired samples are distributed symetrically around zero.
If we reject the null hypothesis, then the paired pre and post operation treatment effects do not have the same underlying distribution, and the operation could have caused a significant difference to the measured treatment effect. And if we fail to reject the null hypothesis, then the paired treatment effect samples could have the same underlying distribution, and the treatment may not have caused a significant difference to the measured treatment effect.


```python
delta_cols = ['NRS', 'MPQ', 'ODI', 'PCS', 'BDI']
for col in delta_cols:
    result_wilcoxon_test = stats.wilcoxon(x=df[f'{col} Pre-Op'], y=df[f'{col} Post-Op'], nan_policy='omit', zero_method='wilcox')

    print(f"Wilcoxon Signed-Rank Test: {col} \nStatistic: {np.round(result_wilcoxon_test.statistic, 3)} \nP-value: {np.round(result_wilcoxon_test.pvalue, 3)}")
    if result_wilcoxon_test.pvalue >  0.05:
        print(f'Pre and Post Op {col} samples could have the same underlying distribution. (fail to reject null hypothesis)')
    else:
        print(f'Pre and Post Op {col} samples do not have the same underlying distribution. (reject null hypothesis)')
    print("********************** \n")
```

    Wilcoxon Signed-Rank Test: NRS
    Statistic: 1.0
    P-value: 0.0
    Pre and Post Op NRS samples do not have the same underlying distribution. (reject null hypothesis)
    **********************

    Wilcoxon Signed-Rank Test: MPQ
    Statistic: 48.0
    P-value: 0.3
    Pre and Post Op MPQ samples could have the same underlying distribution. (fail to reject null hypothesis)
    **********************

    Wilcoxon Signed-Rank Test: ODI
    Statistic: 10.0
    P-value: 0.003
    Pre and Post Op ODI samples do not have the same underlying distribution. (reject null hypothesis)
    **********************

    Wilcoxon Signed-Rank Test: PCS
    Statistic: 44.0
    P-value: 0.074
    Pre and Post Op PCS samples could have the same underlying distribution. (fail to reject null hypothesis)
    **********************

    Wilcoxon Signed-Rank Test: BDI
    Statistic: 45.5
    P-value: 0.09
    Pre and Post Op BDI samples could have the same underlying distribution. (fail to reject null hypothesis)
    **********************



Like with the Student's T-Test, we reject the [null hypothesis](#h3) for treatment effect Numerical Rating Scale (NRS), we also reject the [null hypothesis](#h3) for the treatment effect Oswestry Disability Index (ODI). This suggests that there is a statistically significant difference between these measured treatment effects from before and after the subjects' operations.

For the other three treatment effects, we fail to reject the [null hypothesis](#h3), and should conclude that there is no significant differences to the measured treatment effects before and after the subjects' operations.

## Conclusions

[We found large, average percent descreases between pre and post operation data across all five measured treatment effects for subject's reported pain, depression, and pain induced disability.](#H1)Using both parametric and non-parametric statistical testing we found there was a statistically significant difference to the subject's pain and disability as reported with the Numerical Rating Scale for pain and the Oswestry Disability Index for lower-back pain induced disability.
