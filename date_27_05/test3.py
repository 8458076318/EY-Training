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

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Titanic Dashboard', fontsize=16, fontweight='bold')

# Panel 1: Survival count
ax1 = axes[0, 0]
sns.countplot(data=df, x='survived', ax=ax1)
ax1.set_title('Survival Count')
ax1.set_xlabel('Survived')

# Panel 2: Survival by class
ax2 = axes[0, 1]
sns.barplot(data=df, x='pclass', y='survived', hue='sex', ax=ax2)
ax2.set_title('Survival Rate by Class and Sex')
ax2.set_xlabel('Passenger Class')

# Panel 3: Age distribution
ax3 = axes[0, 2]
sns.histplot(data=df, x='age', hue='survived', bins=30, ax=ax3)
ax3.set_title('Age Distribution by Survival')

# Panel 4: Fare vs Age scatter
ax4 = axes[1, 0]
colors = {1: '#1976D2', 2: '#43A047', 3: '#E53935'}
for cls in [1, 2, 3]:
    subset = df[df['pclass'] == cls]
    ax4.scatter(subset['age'], subset['fare'], c=colors[cls],
                alpha=0.5, s=40, label=str(cls))
ax4.set_yscale('log')
ax4.set_title('Fare vs Age by Class')
ax4.legend(title='Class')

# Panel 5 (YOUR TURN): Countplot by age_group coloured by sex
ax5 = axes[1, 1]
sns.countplot(data=df, x='age_group', hue='sex', ax=ax5)
ax5.set_title('Passengers by Age Group and Sex')
ax5.set_xlabel('Age Group')
ax5.set_ylabel('Count')
ax5.tick_params(axis='x', rotation=15)

# Hide unused 6th panel
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig('titanic_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()