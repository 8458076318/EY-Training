import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Notebook display settings
# %matplotlib inline

plt.rcParams['figure.dpi'] = 120

sns.set_theme(style='whitegrid', palette='muted')

# Load dataset
df = sns.load_dataset('titanic').dropna(subset=['age', 'embarked'])

df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 120],
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

print('Dataset shape:', df.shape)
print(plt.show())

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='age_group', hue='survived')
plt.title('Survival Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# Age distribution
sns.histplot(data=df, x='age', hue='survived', bins=30)
plt.show()

# Survival by sex and age group
sns.catplot(data=df, x='age_group', hue='sex', col='survived', kind='count')
plt.show()

# Part 3 : Scatter plot
fig, ax = plt.subplots(figsize=(7, 5))
colors = {1: '#1976D2', 2: '#43A047', 3: '#E53935'}
labels = {1: '1st class', 2: '2nd class', 3: '3rd class'}

for cls in [1, 2, 3]:
    subset = df[df['pclass'] == cls]
    ax.scatter(subset['age'], subset['fare'], c=colors[cls], alpha=0.5,
               s=40, edgecolors='white', linewidth=0.3, label=labels[cls])

ax.set_title('Fare vs age by passenger class')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Ticket fare (£)')
ax.legend(title='Class')
ax.set_yscale('log')

# Annotate an outlier
max_fare_idx = df['fare'].idxmax()
ax.annotate(f"Highest fare: £{df.loc[max_fare_idx, 'fare']:.0f}",
            xy=(df.loc[max_fare_idx, 'age'], df.loc[max_fare_idx, 'fare']),
            xytext=(50, 400),
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.show()

