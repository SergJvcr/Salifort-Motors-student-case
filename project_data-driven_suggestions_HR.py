# Working with data
import numpy as np
import pandas as pd
# Creating visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# Displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)
# Splitting data into subsets and tuning params (search the best params)
from sklearn.model_selection import train_test_split, GridSearchCV
# Analyzing models: how they perfomance itself
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.tree import plot_tree, DecisionTreeClassifier # 1 - plotting a tree, 2 - modelling: Decision Tree Model
# Modelling data
from xgboost import XGBClassifier # XGBoost models
from sklearn.ensemble import RandomForestClassifier # Random Forest model
from sklearn.linear_model import LogisticRegression # Logistic Regression model
# Saving the final results of modelling
import pickle # optional part

# Load dataset into a dataframe
df0 = pd.read_csv("google_data_analitics\\HR_capstone_dataset.csv")

# Display first few rows of the dataframe
print(df0.head(10))

# DATA EXPLORATION (Initial EDA and data cleaning)--------------------------------------------------
# Gather basic information about the data
print(df0.info())

print(f'The dataset contains {df0.shape[0]} rows and {df0.shape[1]} columns.')

# Gather descriptive statistics about the data
print(df0.describe(include='all'))

# Rename columns
print(df0.columns)

df0 = df0.rename(columns={'promotion_last_5years': 'promotion_last_5_years', 
                          'Department': 'department', 
                          'Work_accident': 'work_accident', 
                          'average_montly_hours': 'average_monthly_hours', 
                          'time_spend_company': 'years_in_company'})

print(df0.columns)

# Check for missing values
print(df0.isna().sum())

# Check for duplicates
print(df0.duplicated().sum())

# Inspect some rows containing duplicates
duplicated_data = df0[df0.duplicated()]
print(duplicated_data.head(10))

# Drop duplicates and save resulting dataframe in a new variable
df_cleaned = df0.drop_duplicates(keep='first')

# Display first few rows of new dataframe
print(df_cleaned.head())

print(f'Here is/are {df_cleaned.duplicated().sum()} duplicated row(s) after cleaning dataset.')
print(f'The cleaned dataset contains {df_cleaned.shape[0]} rows.')

# Check outliers
# Create a boxplot to visualize distribution of `years_in_company` and detect any outliers
plt.figure(figsize=(8, 2))
plt.title('The distribution of years_in_company feature', fontsize=14)
plt.xticks(fontsize=12, color='black')
plt.xlabel(xlabel='years_in_company', fontsize=12, color='black')
sns.boxplot(data=df_cleaned, x='years_in_company', color='orange')
plt.show()

# Determine the number of rows containing outliers
# Calculating the lower threshold for outliers
# Calculate 25th percentile of years_in_company (the 1st quartile)
percentile_25 = df_cleaned['years_in_company'].quantile(0.25)
# Calculate 75th percentile of years_in_company (the 3rd quartile)
percentile_75 = df_cleaned['years_in_company'].quantile(0.75)
# Calculate interquartile range
iqr = percentile_75 - percentile_25
# Calculate upper and lower thresholds for outliers
upper_limit = percentile_75 + 1.5 * iqr
lower_limit = percentile_25 - 1.5 * iqr
# Show the calculated limits
print(f'The lower limit for years_in_company is {lower_limit}')
print(f'The upper limit for years_in_company is {upper_limit}')
# Taking the subset of outliers
mask_outliers = (df_cleaned['years_in_company'] < lower_limit) | (df_cleaned['years_in_company'] > upper_limit)
ouliers_subset = df_cleaned[mask_outliers]
print(f'The number of outliers in years_in_company feature is {len(ouliers_subset)}')

# Prepare dataset for models that are sensitive to outliers (regression models are sensitive for outliers)
mask_delete_outliers = (df_cleaned['years_in_company'] >= lower_limit) & (df_cleaned['years_in_company'] <= upper_limit)
df_cleaned_without_outliers = df_cleaned[mask_delete_outliers] # deal with only the outliers in years_in_company feature/column

# Check for outliers after cleaning
plt.figure(figsize=(8, 2))
plt.title('The distribution of years_in_company feature after delete outliers', fontsize=14)
plt.xticks(fontsize=12, color='black')
plt.xlabel(xlabel='years_in_company', fontsize=12, color='black')
sns.boxplot(data=df_cleaned_without_outliers, x='years_in_company', color='orange')
plt.show()

# ANALYZE STAGE (continue EDA)--------------------------------------------------
# Get numbers of people who left vs. stayed
print(df_cleaned['left'].value_counts())
print(f"Number of people who stayed in/with the company is {df_cleaned['left'].value_counts()[0]}")
print(f"Number of people who left the company is {df_cleaned['left'].value_counts()[1]}")
print('----------')
# Get percentages of people who left vs. stayed (we got the unbalanced dataset)
print(df_cleaned['left'].value_counts(normalize=True))
print(f"Percent of people who stayed in/with the company is {round(df_cleaned['left'].value_counts(normalize=True)[0] * 100, 2)}%")
print(f"Percent of people who left the company is {round(df_cleaned['left'].value_counts(normalize=True)[1] * 100, 2)} %")

# Data visualizations-----------------------------------------------------------
# Visualizing relationships between variables in the data

# The histigram of number of projects, colours - left/stayed
plt.figure(figsize=(8, 5))
plt.title('The histigram of number of projects')
sns.histplot(data=df_cleaned, x='number_project', hue='left',
             multiple='dodge', 
             element='bars', 
             shrink=4, 
             legend=True, 
             palette=['lightblue', 'orange'])
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Number of projects', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
plt.legend(labels=['left','stayed'], loc='upper right')
plt.show()

print('The counts of employees and amount of their projects', df_cleaned['number_project'].value_counts())
employees_two_projects = df_cleaned[(df_cleaned['number_project'] == 2)]['left'].value_counts(normalize=True)
print(f'The percent of employees with 2 projects who left the company is {round(employees_two_projects[1] * 100, 1)}%')
print(f'The percent of employees with 2 projects who stayed in the company is {round(employees_two_projects[0] * 100, 1)}%')
employees_three_projects = df_cleaned[(df_cleaned['number_project'] == 3)]['left'].value_counts(normalize=True)
print(f'The percent of employees with 3 projects who left the company is {round(employees_three_projects[1] * 100, 1)}%')
print(f'The percent of employees with 3 projects who stayed in the company is {round(employees_three_projects[0] * 100, 1)}%')
employees_four_projects = df_cleaned[(df_cleaned['number_project'] == 4)]['left'].value_counts(normalize=True)
print(f'The percent of employees with 4 projects who left the company is {round(employees_four_projects[1] * 100, 1)}%')
print(f'The percent of employees with 4 projects who stayed in the company is {round(employees_four_projects[0] * 100, 1)}%')
employees_five_projects = df_cleaned[(df_cleaned['number_project'] == 5)]['left'].value_counts(normalize=True)
print(f'The percent of employees with 5 projects who left the company is {round(employees_five_projects[1] * 100, 1)}%')
print(f'The percent of employees with 5 projects who stayed in the company is {round(employees_five_projects[0] * 100, 1)}%')
employees_six_projects = df_cleaned[(df_cleaned['number_project'] == 6)]['left'].value_counts(normalize=True)
print(f'The percent of employees with 6 projects who left the company is {round(employees_six_projects[1] * 100, 1)}%')
print(f'The percent of employees with 6 projects who stayed in the company is {round(employees_six_projects[0] * 100, 1)}%')
employees_seven_projects = df_cleaned[(df_cleaned['number_project'] == 7)]['left'].value_counts(normalize=True)
print(f'The percent of employees with 7 projects who left the company is {round(employees_seven_projects[1] * 100, 1)}%')
print(f'The percent of employees with 7 projects who stayed in the company is 0.0%')

# Or we can do the same thing by building a function and scatter plot
def percent_left_num_projects(dataset):
  '''
  Getting the percent who left the company and their number of projects.
  '''
  list_num_pojects = (dataset['number_project'].unique()).tolist()

  df = pd.DataFrame(columns=['number_of_projects', 'percent_left'])

  for element in list_num_pojects:
    el = dataset[(dataset['number_project'] == element)]['left'].value_counts(normalize=True)
    new_row = pd.DataFrame({'number_of_projects':element, 'percent_left':[round(el[1] * 100, 2)]})
    df = pd.concat([df, new_row], ignore_index=True)
  
  df = df.sort_values(by=['percent_left'], ascending=False).reset_index(drop=True)
  
  return df

df_plot_left_num_projects = percent_left_num_projects(df_cleaned_without_outliers)
print(df_plot_left_num_projects)

plt.figure(figsize=(8, 5))
plt.title('Number of projects VS Percent who left')
sns.scatterplot(data=df_plot_left_num_projects, x='percent_left', y='number_of_projects', color='#7EA310')
plt.xlabel('Employees who left the company, %')
plt.ylabel('Number of projects')
plt.xticks(ticks=np.arange(0, 101, step=10))
plt.grid(axis='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.show()

# The histigram of average number of hours per month, colours - left/stayed
plt.figure(figsize=(15, 5))
plt.title('The histigram of average number of hours per month')
sns.histplot(data=df_cleaned, x='average_monthly_hours', hue='left', 
             multiple='dodge', 
             element='bars', 
             shrink=0.7, 
             legend=True, 
             palette=['lightblue', 'orange'])
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Average number of hours per month', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
plt.axvline(x=160, color='red', label='160 hours/month', ls='--', linewidth=3)
plt.legend(labels=['160 hours/month', 'left', 'stayed'], loc='upper right')
plt.show()

# The histigram of satisfaction level
plt.figure(figsize=(17, 5))
plt.title('The histigram of satisfaction level')
sns.histplot(data=df_cleaned, x='satisfaction_level', 
             multiple='dodge', 
             element='bars', 
             shrink=0.7, 
             legend=True, 
             color='purple')
plt.xticks(np.arange(0, 1.05, step=0.05), fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='The level of satisfaction', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
average_satisfaction = df_cleaned['satisfaction_level'].mean()
plt.axvline(x=average_satisfaction, color='red', label='0.63', ls='--', linewidth=3)
plt.legend(labels=['0.63 - av. of satisfaction'], loc='upper right')
plt.show()

# The histigram of satisfaction level (last evaluation)
plt.figure(figsize=(17, 5))
plt.title('The histigram of satisfaction level (last evaluation)')
sns.histplot(data=df_cleaned, x='last_evaluation', 
             multiple='dodge', 
             element='bars', 
             shrink=0.7, 
             legend=True, 
             color='pink')
plt.xticks(np.arange(0, 1.05, step=0.05), fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='The level of satisfaction', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
average_satisfaction = df_cleaned['last_evaluation'].mean()
plt.axvline(x=average_satisfaction, color='red', label='0.72', ls='--', linewidth=3)
plt.legend(labels=['0.72 - av. of satisfaction'], loc='upper right')
plt.show()

# The histigram of satisfaction level (stayed VS left)
plt.figure(figsize=(12,8))
plt.title('The histigram of satisfaction level (stayed VS left)')
sns.histplot(data=df_cleaned, x='satisfaction_level', hue='left', 
             multiple='dodge', 
             element='bars', 
             shrink=0.7, 
             legend=True, 
             palette=['lightblue', 'orange'])
plt.xticks(np.arange(0, 1.05, step=0.05), fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='The level of satisfaction', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
average_satisfaction = df_cleaned['satisfaction_level'].mean()
plt.axvline(x=average_satisfaction, color='red', label='0.63', ls='--', linewidth=3)
plt.legend(labels=['0.63 - av. of satisfaction', 'left', 'stayed'], loc='upper right')
plt.show()

# The histigram of satisfaction level VS years in company
left_satisfaction_years_in = df_cleaned[df_cleaned['left'] == 1]
plt.figure(figsize=(17, 5))
plt.title('The histigram of satisfaction level VS years in company')
sns.scatterplot(data=left_satisfaction_years_in, x='satisfaction_level', y='years_in_company',
                 legend=True, 
                 alpha=0.8, 
                 color='red')
plt.xticks(np.arange(0, 1.1, step=0.1), fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='The level of satisfaction', fontsize=12, color='black')
plt.ylabel(ylabel='Years in company', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.axvline(x=0.35, color='blue', ls='--', linewidth=3)
plt.axvline(x=0.52, color='blue', ls='--', linewidth=3)
plt.axvline(x=0.72, color='green', ls='--', linewidth=3)
plt.axvline(x=0.92, color='green', ls='--', linewidth=3)
plt.show()

# The histigram of years in company VS salary (0-6 years)
short_time_in_company = df_cleaned[df_cleaned['years_in_company'] < 7]
long_time_in_company = df_cleaned[df_cleaned['years_in_company'] > 6]
fig, ax = plt.subplots(1, 2, figsize = (22,8))
ax[0].set_title('The histigram of years in company VS salary (0-6 years)')
sns.histplot(data=short_time_in_company, x='years_in_company', 
             hue='salary', hue_order=['low', 'medium', 'high'], 
             legend=True, 
             multiple='dodge', 
             element='bars', 
             shrink=5, 
             palette=['red', 'orange', 'green'], 
             ax=ax[0])
ax[0].set_xlabel('Years in company', fontsize=12, color='black')

ax[1].set_title('The histigram of years in company VS salary (7-10 years)')
sns.histplot(data=long_time_in_company, x='years_in_company',  
             hue='salary', hue_order=['low', 'medium', 'high'], 
             legend=True, 
             multiple='dodge', 
             element='bars', 
             shrink=1, 
             palette=['red', 'orange', 'green'], 
             ax=ax[1])
ax[1].set_xlabel('Years in company', fontsize=12, color='black')
ax[1].set_xticks(np.arange(7, 11, step=1))

plt.legend(labels=['high', 'medium', 'low'], loc='upper right')
plt.show()

# The histigram of salary, colours - left/stayed
plt.figure(figsize=(5, 5))
plt.title('The histigram of salary')
sns.histplot(data=df_cleaned, x='salary', hue='left', multiple='dodge', element='bars', shrink=0.7, legend=True, palette=['lightgreen', 'orange'])
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Salary', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.show()

print(df_cleaned['salary'].value_counts())
employees_low_salary = df_cleaned[(df_cleaned['salary'] == 'low')]['left'].value_counts(normalize=True)
print(f'The percent of employees who have low salary and left the company is {round(employees_low_salary[1] * 100, 1)}%')
print(f'The percent of employees who have low salary and stayed in the company is {round(employees_low_salary[0] * 100, 1)}%')
employees_medium_salary = df_cleaned[(df_cleaned['salary'] == 'medium')]['left'].value_counts(normalize=True)
print(f'The percent of employees who have medium salary and left the company is {round(employees_medium_salary[1] * 100, 1)}%')
print(f'The percent of employees who have low medium and stayed in the company is {round(employees_medium_salary[0] * 100, 1)}%')
employees_high_salary = df_cleaned[(df_cleaned['salary'] == 'high')]['left'].value_counts(normalize=True)
print(f'The percent of employees who have high salary and left the company is {round(employees_high_salary[1] * 100, 1)}%')
print(f'The percent of employees who have high salary and stayed in the company is {round(employees_high_salary[0] * 100, 1)}%')

# The histigram of promotion during the last 5 years
plt.figure(figsize=(5, 10))
plt.title('The histigram of promotion during the last 5 years')
sns.histplot(data=df_cleaned, x='promotion_last_5_years', hue='left',
              multiple='dodge', 
              element='bars', 
              shrink=5, 
              legend=True, 
              palette=['lightgreen', 'orange'])
plt.xticks(np.arange(0, 2, step=1), fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='S', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.show()

employees_no_promotion = df_cleaned[(df_cleaned['promotion_last_5_years'] == 0)]['left'].value_counts(normalize=True)
print(f'The percent of employees who have not promotion and left the company is {round(employees_no_promotion[1] * 100, 1)}%')
print(f'The percent of employees who have not promotion and stayed in the company is {round(employees_no_promotion[0] * 100, 1)}%')
employees_promotion = df_cleaned[(df_cleaned['promotion_last_5_years'] == 1)]['left'].value_counts(normalize=True)
print(f'The percent of employees who have promotion and left the company is {round(employees_promotion[1] * 100, 1)}%')
print(f'The percent of employees who have promotion and stayed in the company is {round(employees_promotion[0] * 100, 1)}%')

# The histigram of department, colours - stayed/left
plt.figure(figsize=(10, 7))
plt.title('The histigram of department')
sns.histplot(data=df_cleaned, x='department', hue='left', multiple='dodge', element='bars', shrink=0.5, legend=True, palette=['lightgreen', 'orange'])
plt.xticks(fontsize=12, color='black', rotation=45)
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Departments', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.show()

def percent_left_departments(dataset):
  '''
  Getting the percent who left the company from different departments.
  '''
  list_departments = (dataset['department'].unique()).tolist()

  df = pd.DataFrame(columns=['department', 'percent_left'])

  for element in list_departments:
    el = dataset[(dataset['department'] == element)]['left'].value_counts(normalize=True)
    new_row = pd.DataFrame({'department':element, 'percent_left':[round(el[1] * 100, 2)]})
    df = pd.concat([df, new_row], ignore_index=True)
  
  df = df.sort_values(by=['percent_left'], ascending=False).reset_index(drop=True)
  
  return df

df_plot_left_department = percent_left_departments(df_cleaned_without_outliers)
print(df_plot_left_department)

plt.figure(figsize=(8, 5))
plt.title('Department VS Percent who left')
sns.scatterplot(data=df_plot_left_department, x='percent_left', y='department', color='darkorange')
plt.xlabel('Employees who left the company, %')
plt.ylabel('Department')
plt.xticks(ticks=np.arange(12, 20, step=0.5))
plt.grid(axis='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.show()

# The histigram of work accident, colours - left/stayed
plt.figure(figsize=(5, 8))
plt.title('The histigram of work accident')
sns.histplot(data=df_cleaned, x='work_accident', hue='left', 
             multiple='dodge', 
             element='bars', 
             shrink=5, 
             legend=True, 
             palette=['lightgreen', 'orange'])
plt.xticks(np.arange(0, 2, step=1), ['No', 'Yes'], fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Work accidents', fontsize=12, color='black')
plt.ylabel(ylabel='Count of employees', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.show()

no_accidents = df_cleaned[(df_cleaned['work_accident'] == 0)]['left'].value_counts(normalize=True)
print(f'The percent of employees without accidents and left the company is {round(no_accidents[1] * 100, 1)}%')
print(f'The percent of employees without department and stayed in the company is {round(no_accidents[0] * 100, 1)}%')
yes_accidents = df_cleaned[(df_cleaned['work_accident'] == 1)]['left'].value_counts(normalize=True)
print(f'The percent of employees with accidents and left the company is {round(yes_accidents[1] * 100, 1)}%')
print(f'The percent of employees with department and stayed in the company is {round(yes_accidents[0] * 100, 1)}%')

# The histigram of salary VS average number of hours per month for left employees
left_employees = df_cleaned[df_cleaned['left'] == 1]
plt.figure(figsize=(17, 4))
plt.title('The histigram of salary VS average number of hours per month for left employees')
sns.scatterplot(data=left_employees, x='average_monthly_hours', y='salary', 
                 legend=True, 
                 alpha=1,
                 color='red')
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Average monthly hours', fontsize=12, color='black')
plt.ylabel(ylabel='Salary', fontsize=12, color='black')
plt.legend(labels=['left'], loc='upper right')
plt.show()

# The histigram of promotion VS average number of hours per month
plt.figure(figsize=(15, 2))
plt.title('The histigram of promotion VS average number of hours per month')
sns.scatterplot(data=left_employees, x='average_monthly_hours', y='promotion_last_5_years', 
                 legend=True, 
                 alpha=1, 
                 color='red')
plt.xticks(fontsize=12, color='black')
plt.yticks([0, 1], ['No', 'Yes'], fontsize=12, color='black')
plt.xlabel(xlabel='Average monthly hours', fontsize=12, color='black')
plt.ylabel(ylabel='Promotion', fontsize=12, color='black')
plt.legend(labels=['left'], loc='upper right')
plt.show()

# The histigram of promotion VS years in company
plt.figure(figsize=(17, 7))
plt.title('The histigram of promotion VS years in company')
sns.histplot(data=left_employees, x='years_in_company', y='promotion_last_5_years', 
             color='red', 
             multiple='dodge', 
             element='bars', 
             legend=True)
plt.xticks(fontsize=12, color='black')
plt.yticks([0, 1], ['No', 'Yes'], fontsize=12, color='black')
plt.xlabel(xlabel='Years in company', fontsize=12, color='black')
plt.ylabel(ylabel='Promotion', fontsize=12, color='black')
plt.legend(labels=['left'], loc='upper right')
plt.show()

# The histigram of number of projects VS average number of hours per month, colours - left/stayed
plt.figure(figsize=(17, 7))
plt.title('The histigram of number of projects VS average number of hours per month')
sns.histplot(data=df_cleaned, x='average_monthly_hours', y='number_project', hue='left', 
             multiple='dodge', 
             element='bars', 
             legend=True, 
             palette=['lightblue', 'red'])
plt.xticks(fontsize=12, color='black')
plt.yticks(np.arange(2, 8),fontsize=12, color='black')
plt.xlabel(xlabel='Average monthly hours', fontsize=12, color='black')
plt.ylabel(ylabel='Number of projects', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.show()

# The histigram of level of satisfaction VS average number of hours per month, colours - left/stayed
plt.figure(figsize=(16, 10))
plt.title('The histigram of level of satisfaction VS average number of hours per month')
sns.scatterplot(data=df_cleaned, x='average_monthly_hours', y='satisfaction_level', hue='left', 
                 alpha=0.2, 
                 legend=True)
plt.xticks(fontsize=12, color='black')
plt.yticks(np.arange(0, 1.05, step=0.1), fontsize=12, color='black')
plt.xlabel(xlabel='Average monthly hours', fontsize=12, color='black')
plt.ylabel(ylabel='Level of satisfaction', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
average_satisfaction = df_cleaned['satisfaction_level'].mean()
plt.hlines(y=average_satisfaction, color='red', xmin=95, xmax=315, ls='--', linewidth=3)
plt.axvline(x=160, color='red', ls='--', linewidth=3)
plt.show()

# The histigram of level of satisfaction VS average number of hours per month
plt.figure(figsize=(16, 10))
plt.title('The histigram of level of satisfaction VS average number of hours per month')
sns.scatterplot(data=left_employees, x='average_monthly_hours', y='satisfaction_level', 
                 alpha=0.2, 
                 legend=True,
                 color='orange')
plt.xticks(fontsize=12, color='black')
plt.yticks(np.arange(0, 1.05, step=0.1), fontsize=12, color='black')
plt.xlabel(xlabel='Average monthly hours', fontsize=12, color='black')
plt.ylabel(ylabel='Level of satisfaction', fontsize=12, color='black')
plt.legend(labels=['left'], loc='upper right')
average_satisfaction = df_cleaned['satisfaction_level'].mean()
plt.hlines(y=average_satisfaction, color='red', xmin=95, xmax=315, ls='--', linewidth=3)
plt.axvline(x=160, color='red', ls='--', linewidth=3)
plt.show()

# Heatmap of correlation of the features
# for_heatmap_subset = df_cleaned.drop(['salary', 'department'], axis=1)
# plt.figure(figsize=(16, 9))
# heatmap = sns.heatmap(for_heatmap_subset, 
#                       vmin=-1, vmax=1, 
#                       annot=True, 
#                       cmap='coolwarm')
# heatmap.set_title('Heatmap of correlation of the features', fontdict={'fontsize':14}, pad=12)
# plt.show()

# The histigram of years in company VS average monthly hours for those employees who LEFT
who_left = df_cleaned[df_cleaned['left'] == 1]
plt.figure(figsize=(12, 9))
plt.title('The histigram of years in company VS average monthly hours for those employees who LEFT')
sns.scatterplot(data=who_left, x='years_in_company', y='average_monthly_hours', color='orange', alpha=1, legend=True)
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.xlabel(xlabel='Years in company', fontsize=12, color='black')
plt.ylabel(ylabel='Average monthly hours', fontsize=12, color='black')
plt.legend(labels=['left', 'stayed'], loc='upper right')
plt.show()

# MODEL BUILDING------------------------------------------------------------------------------------------------
print(df_cleaned_without_outliers.head(5))

# Binary Logistic Regression Model------------------------------------------------------------------------------
# The 1st step is cleaned data from outliers and convert non-numeric (department, salary) data to numeric type
# The data already cleaned from outliers (df_cleaned_df_cleaned_without_outliers)
df_prepared_logistic_regression = df_cleaned_without_outliers.copy()
# Ordinary encode the salary column (values have hierarchy in this column)
df_prepared_logistic_regression['salary'] = pd.factorize(df_prepared_logistic_regression['salary'], sort=False)[0] + 1
# Dammy encode the department column (values are equal in this column)
df_prepared_logistic_regression = pd.get_dummies(df_prepared_logistic_regression, drop_first=False)
print(df_prepared_logistic_regression.head(5))

plt.figure(figsize=(16, 16))
sns.heatmap(df_prepared_logistic_regression.corr(), vmin=-1, vmax=1, annot=True, cmap="coolwarm")
plt.title('Heatmap of the dataset')
plt.show()

# The 2nd step is isolate (outcome/target) y-variable and select features for X-variable
y = df_prepared_logistic_regression['left']
X = df_prepared_logistic_regression.drop(['left'], axis=1)

# The 3rd step - divide X and y variables into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
print(f'The size of the X_train is {X_train.shape[0]} rows, {round(X_train.shape[0] / df_prepared_logistic_regression.shape[0] * 100, 2)}%')
print(f'The size of the X_test is {X_test.shape[0]} rows, {round(X_test.shape[0] / df_prepared_logistic_regression.shape[0] * 100, 2)}%')
print(f'The size of the y_train is {y_train.shape[0]} rows, {round(y_train.shape[0] / df_prepared_logistic_regression.shape[0] * 100, 2)}%')
print(f'The size of the y_train is {y_test.shape[0]} rows, {round(y_test.shape[0] / df_prepared_logistic_regression.shape[0] * 100, 2)}%')

# The 4th step is building the model
# Fit a LogisticRegression model to the data
logistic_classifier = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)

# The 5th step is testing model
y_predicted_log_reg = logistic_classifier.predict(X_test)

# Analyze the results
print(f"Accuracy: {round(accuracy_score(y_test, y_predicted_log_reg), 3)}")
print(f"Precision: {round(precision_score(y_test, y_predicted_log_reg), 3)}")
print(f"Recall: {round(recall_score(y_test, y_predicted_log_reg), 3)}")
print(f"F1 Score: {round(f1_score(y_test, y_predicted_log_reg), 3)}")

# Create a confusion matrix
log_conf_matrix = confusion_matrix(y_test, y_predicted_log_reg, labels=logistic_classifier.classes_)
log_display = ConfusionMatrixDisplay(confusion_matrix=log_conf_matrix, display_labels=logistic_classifier.classes_)
log_display.plot(values_format='')
plt.title('Confusion matrix: the results of Logistic Regression Model')
plt.show()

# ROC curve
print(f'The AUC value is {roc_auc_score(y_test, y_predicted_log_reg)}')
RocCurveDisplay.from_predictions(y_test, y_predicted_log_reg, plot_chance_level=True, color='orange')
plt.title('ROC curve: \n Features VS Left')
plt.show()

left_column_balance = df_prepared_logistic_regression['left'].value_counts(normalize=True)
print(f'The percent who stayed is {round(left_column_balance[0] * 100, 1)}%')
print(f'The percent who left is {round(left_column_balance[1] * 100, 1)}%')
# Dataset is imbalanced but not the worst case (not too much)

# Create classification report for logistic regression model
target_names = ['Predicted whi would not leave', 'Predicted who would leave']
print(classification_report(y_test, y_predicted_log_reg, target_names=target_names))

# The classification report above shows that the logistic regression model achieved 
# a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. 
# However, if it's most important to predict employees who leave, then the scores are significantly lower.

# Decision Tree Model------------------------------------------------------------------------------
# Cross-validating grid-searching to exhuastively search for the best model parameters
tree = DecisionTreeClassifier(random_state=42)
cv_params = {'max_depth':[2, 4, 6, 8, None],
             'min_samples_leaf': [1, 2, 5],
             'min_samples_split': [2, 4, 6]
             }
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

decision_tree = GridSearchCV(estimator=tree, 
                             param_grid=cv_params, 
                             scoring=scoring, 
                             cv=5, 
                             refit='roc_auc')
decision_tree.fit(X_train, y_train)

# Get the best parameters
print('The best parameters for decision tree:')
print(decision_tree.best_params_)
# Check best AUC score on CV
print(f'The best AUC score on CV is {round(decision_tree.best_score_, 3)}')

# This is a strong AUC score, which shows that this model can predict employees who will leave very well

def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table

# Prediction of the data using the trained model
decision_tree_cv_results = make_results('decision tree CV', decision_tree, 'auc')
print(decision_tree_cv_results)

# Testing Decision Tree Model
# Get scores on test data
y_predicted_test_decision_tree = decision_tree.best_estimator_.predict(X_test)

def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
    model_name (string): Your choice: how the model will be named in the output table
    preds: numpy array of test predictions
    y_test_data: numpy array of y_test data

    Out:
    table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    auc = roc_auc_score(y_test_data, preds)
    
    table = pd.DataFrame({'model': [model_name],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy], 
                        'auc': [auc]
                        })

    return table

# Get scores on test data
decision_tree_cv_testing = get_test_scores('decision tree CV TEST', y_predicted_test_decision_tree, y_test)
decision_tree_cv_results = pd.concat([decision_tree_cv_results, decision_tree_cv_testing], axis=0)
print(decision_tree_cv_results)

# Visualisation with confusion matrix for testing data of the model
conf_matrix = confusion_matrix(y_test, y_predicted_test_decision_tree, labels=decision_tree.classes_)
display_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=decision_tree.classes_)
display_conf_matrix.plot(values_format='') # `values_format=''` suppresses scientific notation
plt.title('Confusion matrix: the results of Decision Tree Model (testing)')
plt.show()

# Plot the Decision Tree Model
names_for_features = list(X.columns)
names_for_classes = ['stayed', 'left']

plt.figure(figsize=(85, 20))
plot_tree(decision_tree.best_estimator_, max_depth=6, fontsize=5, 
          feature_names=names_for_features, class_names=names_for_classes, filled=True)
plt.show()

# Get the relative importances of each feature that was used to buid this model
importances = decision_tree.best_estimator_.feature_importances_
decision_tree_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
decision_tree_importances.plot(color='lightseagreen', kind='barh', figsize=(10, 6), legend=True)
plt.title('The relative importances of each features')
plt.legend(loc='lower right', labels=['Importance'])
plt.show()

# Random Forest Model------------------------------------------------------------------------------
# Cross-validating grid-searching to exhuastively search for the best model parameters
rf = RandomForestClassifier(random_state=42)
cv_params = {'max_depth': [3, 5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [2, 3, 4],
             'n_estimators': [300, 500],
             }
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

random_forest = GridSearchCV(estimator=rf, 
                             param_grid=cv_params, 
                             scoring=scoring, 
                             cv=5, 
                             refit='roc_auc', 
                             verbose=1)
random_forest.fit(X_train, y_train)

# Get the best parameters
print('The best parameters for random forest:')
print(random_forest.best_params_)
# Check best AUC score on CV
print(f'The best AUC score on CV is {round(random_forest.best_score_, 3)}')

random_forest_cv_results = make_results('random forest CV', random_forest, 'auc')
decision_tree_cv_results = pd.concat([decision_tree_cv_results, random_forest_cv_results], axis=0)
print(decision_tree_cv_results)

# Testing the model and saving the result
y_predicted_test_random_forest = random_forest.best_estimator_.predict(X_test)
random_forest_cv_testing = get_test_scores('random forest CV TEST', y_predicted_test_random_forest, y_test)
decision_tree_cv_results = pd.concat([decision_tree_cv_results, random_forest_cv_testing], axis=0)
print(decision_tree_cv_results)

# Visualisation with confusion matrix for testing data of the model
conf_matrix = confusion_matrix(y_test, y_predicted_test_random_forest, labels=random_forest.classes_)
display_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=random_forest.classes_)
display_conf_matrix.plot(values_format='') # `values_format=''` suppresses scientific notation
plt.title('Confusion matrix: the results of Random Forest Model (testing)')
plt.show()

# Get the relative importances of each feature that was used to buid this model
importances = random_forest.best_estimator_.feature_importances_
random_forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
random_forest_importances.plot(color='pink', kind='barh', figsize=(10, 6), legend=True)
plt.title('The relative importances of each features')
plt.legend(loc='lower right', labels=['Importance'])
plt.show()

# XGBoost Model------------------------------------------------------------------------------
xgb = XGBClassifier(objective='binary:logistic', random_state=42)
# Define parameters for tuning
cv_params = {'max_depth':[2, 4, 6],
             'min_child_weight':[3, 5],
             'learning_rate':[0.1, 0.2, 0.3],
             'n_estimators':[5, 10, 15],
             'subsample':[0.7],
             'colsample_bytree':[0.7]
            }
# Define criterias as `scoring`
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# Construct GridSearch cross-validation
xgb_classifier = GridSearchCV(estimator=xgb, 
                              param_grid=cv_params, 
                              scoring=scoring, 
                              cv=5, 
                              refit='roc_auc')
# fit the GridSearch model to training data
xgb_classifier.fit(X_train, y_train)

print('The best parameters for XGBosst model:')
print(xgb_classifier.best_params_)
# Check best AUC score on CV
print(f'The best AUC score on CV is {round(xgb_classifier.best_score_, 3)}')

xgb_classifier_cv_results = make_results('XGBoost CV', xgb_classifier, 'auc')
decision_tree_cv_results = pd.concat([decision_tree_cv_results, xgb_classifier_cv_results], axis=0)
print(decision_tree_cv_results)

# Testing the model and saving the result
y_predicted_test_xgb_classifier = xgb_classifier.best_estimator_.predict(X_test)
xgb_classifier_cv_testing = get_test_scores('XGBoost CV TEST', y_predicted_test_xgb_classifier, y_test)
decision_tree_cv_results = pd.concat([decision_tree_cv_results, xgb_classifier_cv_testing], axis=0)
decision_tree_cv_results

# Visualisation with confusion matrix for testing data of the model
conf_matrix = confusion_matrix(y_test, y_predicted_test_xgb_classifier, labels=xgb_classifier.classes_)
display_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=xgb_classifier.classes_)
display_conf_matrix.plot(values_format='') # `values_format=''` suppresses scientific notation
plt.title('Confusion matrix: the results of XGBoost Model (testing)')
plt.show()

# Get the relative importances of each feature that was used to buid this model
importances = xgb_classifier.best_estimator_.feature_importances_
xgb_classifier_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
xgb_classifier_importances.plot(color='coral', kind='barh', figsize=(10, 6), legend=True)
plt.title('The relative importances of each features')
plt.legend(loc='lower right', labels=['Importance'])
plt.show()
