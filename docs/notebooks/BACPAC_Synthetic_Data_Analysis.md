---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# BACPAC Synthetic Data Analysis

## Qiong Liu
### April 2nd, 2021

In this Jupyter notebook, we used the [BACPAC study](https://healdata.org/portal/discovery/HDP00258/) as an example to demonstrate how to navigate datasets within the workspace in HEAL and conduct data analysis using Python libraries.

## Table of Content
- [Set up notebook](#h1)
- [Pull file objects using the Gen3 SDK](#Pull-file-objects-using-the-Gen3-SDK)
- [Demographic characteristics of participants in BACPAC](#Demographic-characteristics-of-participants-in-BACPAC)
- [Opiod pain medication profiling at two time points](#Opiod-pain-medication-profiling-at-two-time-points)
- [Physical function outcomes](#Physical-function-outcomes)

+++

(h1)=
## Set up notebook

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import numpy as np
import json
import requests
import os
plotly.offline.init_notebook_mode()
```

## Query study metadata
Users can query [study metadata in HEAL data commons](https://healdata.org/mds/metadata?data=True) using our metadata service (MDS). The cell below shows how to retrieve the metadata of the BACPAC study by interacting with the gen3 MDS endpoint.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Query the metadata of BACPAC using the project number "1U24AR076730-01"
response=requests.get("https://healdata.org/mds/metadata?data=True&limit=1000&gen3_discovery.project_number=1U24AR076730-01")
metadata_text=response.text
metadata_object=json.loads(metadata_text)
meta_df = pd.json_normalize([sub['gen3_discovery'] for sub in metadata_object.values() if 'gen3_discovery' in sub.keys()])
meta_df[['research_focus_area', 'study_description_summary', 'institutions']].transpose()
```

## Pull file objects using the Gen3 SDK

```{code-cell} ipython3
!gen3 drs-pull object dg.H34L/80f0a338-18e0-48de-b70f-cdabd63f67d9
!gen3 drs-pull object dg.H34L/530fd95c-48b6-488e-a699-9377180bd82d
!gen3 drs-pull object dg.H34L/654d7f1f-b61c-49a9-8a74-c82400fa4c27
```

## Demographic characteristics of participants in BACPAC

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Read the demographic tsv file into dataframe
demo_bacpac=pd.read_csv("./participant_SMART.tsv", sep="\t", encoding="utf-8")


# Define age groups within participants
age_list = list(demo_bacpac["age_in_years"])
def age_group(agelist):
    min_age = min(agelist)
    grouplabel1 = str(min_age) + "-55 yr"
    grouplabel2= ">55 yr"
    grouplist = []
    for i in agelist:
        if i <=55:
            grouplist.append(grouplabel1)
        else:
            grouplist.append(grouplabel2)
    return grouplist

agegrouplist = age_group(age_list)
demo_bacpac["age_group"] = agegrouplist

# Compute three frequency tables using demographic factors
df1=pd.crosstab(index=demo_bacpac['race'], columns=demo_bacpac['sex'])
df2=pd.crosstab(index=demo_bacpac['ethnicity'], columns=demo_bacpac['sex'])
df3=pd.crosstab(index=demo_bacpac['age_group'], columns=demo_bacpac['sex'])

# Dsiplay concatenated tables
pd.concat([df1, df2, df3], keys=['race', 'ethnicity', 'age_group'])
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Generate a stacked bar chart of participants in BACPAC
new_df2 = pd.DataFrame(df2.stack())
new_df2.reset_index(inplace=True)
new_df2 = new_df2.rename({0:"Count", "sex": "Sex", "ethnicity": "Ethnicity"}, axis="columns")

fig1 = px.bar(new_df2, x="Sex", y="Count", color="Ethnicity",
             title= "Ethnicity and Sex Characteristics of Participants in the BACPAC Study",
             width= 800, height = 500)
fig1.update_layout(title_font_size=20)
fig1.show()
```

## Opiod pain medication profiling at two time points

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Read substance use tsv file into dataframe
substance_df = pd.read_csv("./substance_use_SMART.tsv", sep="\t", encoding="utf-8")

# Combine substance use df and demographic df based on participant id
def find_participant(mydf, endstr):
    participant_id = []
    for i in list(mydf["submitter_id"]):
        i_participant = i.rstrip(endstr)
        participant_id.append(i_participant)
    return participant_id
substance_participant_id = find_participant(substance_df,"_sc")
substance_df["participant_id"] = substance_participant_id
demo_combine_substance = substance_df.merge(demo_bacpac, left_on="participant_id",
                                            right_on="submitter_id", how="outer")

# Add one property of time point in the df
def find_timepoint(mydf):
    timepoint = []
    for i in list(mydf["visits.submitter_id"]):
        if i.endswith("Week 0"):
            timepoint.append("Week 0")
        else:
            timepoint.append("Week 12")
    return timepoint

demo_combine_substance["time_point"] = find_timepoint(demo_combine_substance)

# Compute a frequency table using opioid medication factor and time point factor
opioid_crosstab = pd.crosstab(index=demo_combine_substance['OPIOID01'],
                              columns=demo_combine_substance['time_point'])
new_opioid = pd.DataFrame(opioid_crosstab.stack())
new_opioid.reset_index(inplace=True)
new_opioid = new_opioid.rename({0:"Count", "OPIOID01": "Taking Opioid", "time_point": "Time Point"},
                               axis="columns")

# Generate a bar chart showing the opioid taking at two time points
fig2 = px.bar(new_opioid, x="Taking Opioid", y="Count", color="Taking Opioid",
             facet_row="Time Point", width=800, height=400)
fig2.update_layout(title_text="Self-Report of Opioid Pain Medication Use at Baseline and Twelve Weeks",title_font_size=20)
for data in fig2.data:
    data["width"]=0.6
fig2.show()
```

- We observed an increase of participants taking opioid pain medication at the week 12 time point compared to baseline.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Generate a bar chart showing the opioid taking at two time points in different sex groups
opioid_gender = pd.crosstab(index=[demo_combine_substance['OPIOID01'], demo_combine_substance['sex']],
                            columns=demo_combine_substance['time_point'])
new_opioid_gender = pd.DataFrame(opioid_gender.stack())
new_opioid_gender.reset_index(inplace=True)
new_opioid_gender = new_opioid_gender.rename({0:"Count", "OPIOID01": "Taking Opioid",
                                              "time_point": "Time Point", "sex": "Sex"}, axis="columns")
fig3 = px.bar(new_opioid_gender, y="Sex", x="Count", color="Taking Opioid",
             facet_col="Time Point", width=800, height=400,  orientation='h',
             category_orders={"Sex": ["Intersex", "Unknown", "Male", "Female"]})
fig3.update_layout(title_text="Opioid Pain Medication at Two Time Points in Different Sex Groups",
                   title_font_size=20)
fig3.show()
```

- We observed an increase of particpants taking opioid medication at week 12 in both male and femal groups compared to baseline week 0.

## Physical function outcomes
 The cell below uses the Physical Function 6b T-Score to display physical function outcomes in different ethnicity groups at week 0 and week 12.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Read physical_function_SMART.tsv into dataframe and merge the df with demographic
function_df = pd.read_csv("./physical_function_SMART.tsv", sep="\t", encoding="utf-16")
function_participant_id = find_participant(function_df, "_pf")
function_df["participant_id"] = function_participant_id
demo_combine_function = function_df.merge(demo_bacpac, left_on="participant_id",
                                          right_on="submitter_id", how="outer")
demo_combine_function["time_point"] = find_timepoint(demo_combine_function)

# Summary table of ROMIS-Physical Function 6b T-Score in different ethnicity groups
ethnicity_PRPF6BT = demo_combine_function[["time_point",
                                           "PRPF6BT",
                                           "ethnicity"]].groupby(['time_point','ethnicity']).describe()
ethnicity_PRPF6BT
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
# Visualize the distribution of Physical Function 6b T-Score
# at two time points for hispanic and non-hispanic ethnicity groups
fig5 = make_subplots(
    rows=2, cols=2,
    specs=[[{"colspan": 2}, None],
           [{}, {}]],
    subplot_titles=("PROMIS-Physical Function 6b T-Score Distribution at Two Time Points","Hispanic or Latino",
                    "Not Hispanic or Latino"))

fig5.add_trace(go.Histogram(x=demo_combine_function[demo_combine_function["time_point"]=="Week 0"]["PRPF6BT"],
                           marker_color='#EB89B5', opacity=0.75, nbinsx=20, name="Week 0"),
               row=1, col=1)
fig5.add_trace(go.Histogram(x=demo_combine_function[demo_combine_function["time_point"]=="Week 12"]["PRPF6BT"],
                           marker_color='#2B6CBE', opacity=0.75, nbinsx=20, name="Week 12"),
               row=1, col=1)
fig5.add_trace(go.Histogram(x=demo_combine_function[(demo_combine_function["time_point"]=="Week 0")&(demo_combine_function["ethnicity"]=="Hispanic or Latino")]["PRPF6BT"],
                           marker_color='#EB89B5', opacity=0.75, nbinsx=20,showlegend=False),
               row=2, col=1)
fig5.add_trace(go.Histogram(x=demo_combine_function[(demo_combine_function["time_point"]=="Week 12")&(demo_combine_function["ethnicity"]=="Hispanic or Latino")]["PRPF6BT"],
                           marker_color='#2B6CBE', opacity=0.75, nbinsx=20,showlegend=False),
               row=2, col=1)
fig5.add_trace(go.Histogram(x=demo_combine_function[(demo_combine_function["time_point"]=="Week 0")&(demo_combine_function["ethnicity"]=="Not Hispanic or Latino")]["PRPF6BT"],
                           marker_color='#EB89B5', opacity=0.75, nbinsx=20,showlegend=False),
               row=2, col=2)
fig5.add_trace(go.Histogram(x=demo_combine_function[(demo_combine_function["time_point"]=="Week 12")&(demo_combine_function["ethnicity"]=="Not Hispanic or Latino")]["PRPF6BT"],
                           marker_color='#2B6CBE', opacity=0.75, nbinsx=20,showlegend=False),
               row=2, col=2)

fig5.update_layout(barmode='overlay', width=800, height=500,legend_title_text='Time Point')
fig5.update_layout(margin=dict(l=20, r=20, t=50, b=20, pad=2))
fig5.update_yaxes(title_text="Count",
                  title_font_size=15, range=[0, 40], row=1, col=1)
fig5.update_xaxes(title_text="PROMIS-Physical Function 6b T-Score",
                  title_font_size=15,
                  range=[29, 49], row=1, col=1)
fig5.update_yaxes(title_text="Count",
                  title_font_size=15, range=[0, 15], row=2, col=1)
fig5.update_xaxes(title_text="PROMIS-Physical Function 6b T-Score",
                  title_font_size=15, range=[29, 49], row=2, col=1)
fig5.update_yaxes(title_text="Count",
                  title_font_size=15, range=[0, 15], row=2, col=2)
fig5.update_xaxes(title_text="PROMIS-Physical Function 6b T-Score",
                  title_font_size=15, range=[29, 49], row=2, col=2)

fig5.show()
```

```{code-cell} ipython3

```
