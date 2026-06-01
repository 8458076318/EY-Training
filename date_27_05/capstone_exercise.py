# ============================================================
# CELL 1 - Setup (run this first)
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# %matplotlib inline
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='whitegrid')
print('Libraries loaded ✅')


# ============================================================
# CELL 2 - Section 1: Load & Inspect
# ============================================================
df = sns.load_dataset('geyser')

# 1. Print shape
print(df.shape)

# 2. Show first rows
df.head()

# 3. Structural info
df.info()

# 4. Numeric summary
df.describe()

# 5. Value counts on categorical column
df['kind'].value_counts()


# ============================================================
# CELL 3 - Section 2: Clean & Prepare
# ============================================================
print('Shape before cleaning:', df.shape)

# Step 1 - Inspect missing values
print(df.isnull().sum())

# Step 2 - Handle missing values
df.dropna(inplace=True)

# Step 3 - Drop irrelevant columns
# All columns are relevant for geyser dataset — nothing dropped

# Step 4 - Engineer new features
df['duration_bucket'] = pd.cut(df['duration'],
                                bins=[0, 2, 3, 3.5, 6],
                                labels=['very_short','short','medium','long'])
df['kind_encoded'] = df['kind'].map({'short': 0, 'long': 1})

print('Shape after cleaning:', df.shape)
df.head()


# ============================================================
# CELL 4 - Section 3: Explore with Statistics
# ============================================================
# NumPy calculations
p25 = np.percentile(df['duration'], 25)
p75 = np.percentile(df['duration'], 75)
corr = np.corrcoef(df['duration'], df['waiting'])[0, 1]
duration_normalized = (df['duration'] - df['duration'].mean()) / df['duration'].std()

# GroupBy aggregation
group_stats = df.groupby('kind').agg(
    mean_duration=('duration', 'mean'),
    mean_waiting=('waiting', 'mean'),
    count=('duration', 'count'),
    std_duration=('duration', 'std')
)
print(group_stats)

# Pivot table
pivot = pd.pivot_table(df,
                       values='waiting',
                       index='kind',
                       columns='duration_bucket',
                       aggfunc='mean')
print(pivot)

# Formatted summary
print(f"\n📊 Dataset Summary:")
print(f"  Total eruptions:         {len(df)}")
print(f"  Correlation (dur/wait):  {corr:.3f}")
print(f"  25th percentile dur:     {p25:.2f} mins")
print(f"  75th percentile dur:     {p75:.2f} mins")
print(f"  Avg wait (long):         {group_stats.loc['long','mean_waiting']:.1f} mins")


# ============================================================
# CELL 5 - Section 4: Chart 1 — Histogram
# ============================================================
# Chart 1 - Histogram
# Question: How is eruption duration distributed?

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['duration'], bins=20, color='steelblue', edgecolor='white')
ax.set_title('Distribution of Eruption Duration')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Number of Eruptions')
plt.tight_layout()
plt.show()


# ============================================================
# CELL 6 - Section 4: Chart 2 — Scatter Plot
# ============================================================
# Chart 2 - Scatter Plot
# Question: Is there a relationship between duration and waiting time?

fig, ax = plt.subplots(figsize=(8, 5))
for kind, color in zip(['short','long'], ['orange','steelblue']):
    subset = df[df['kind'] == kind]
    ax.scatter(subset['duration'], subset['waiting'],
               label=kind, alpha=0.6, color=color)
ax.set_title('Eruption Duration vs Waiting Time')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Waiting Time (minutes)')
ax.legend(title='Eruption Type')
plt.tight_layout()
plt.show()


# ============================================================
# CELL 7 - Section 4: Chart 3 — Box Plot with Annotation
# ============================================================
# Chart 3 - Box Plot with annotation
# Question: How does waiting time differ between eruption types?

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='kind', y='waiting', palette='Set2', ax=ax)
overall_mean = df['waiting'].mean()
ax.axhline(overall_mean, color='red', linestyle='--', linewidth=1.5)
ax.annotate(f'Overall mean: {overall_mean:.1f} min',
            xy=(0.5, overall_mean),
            xytext=(0.6, overall_mean + 5),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontsize=10)
ax.set_title('Waiting Time by Eruption Type')
ax.set_xlabel('Eruption Type')
ax.set_ylabel('Waiting Time (minutes)')
plt.tight_layout()
plt.show()


# ============================================================
# CELL 8 - Section 4: Chart 4 — Multi-panel Subplot
# ============================================================
# Chart 4 - Multi-panel figure
# Questions: How does duration and waiting time compare across types?

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Geyser Eruption Patterns by Type', fontsize=13)

ax = axes[0]
group_stats['mean_duration'].plot(kind='bar', ax=ax,
                                   color=['orange','steelblue'],
                                   edgecolor='white')
ax.set_title('Panel A: Average Duration by Type')
ax.set_xlabel('Eruption Type')
ax.set_ylabel('Mean Duration (minutes)')
ax.tick_params(axis='x', rotation=0)

ax = axes[1]
group_stats['mean_waiting'].plot(kind='bar', ax=ax,
                                  color=['orange','steelblue'],
                                  edgecolor='white')
ax.set_title('Panel B: Average Waiting Time by Type')
ax.set_xlabel('Eruption Type')
ax.set_ylabel('Mean Waiting Time (minutes)')
ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# CELL 9 - BONUS: Dashboard saved as PNG
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Old Faithful Geyser — Analysis Dashboard', fontsize=15)

axes[0,0].hist(df['duration'], bins=20, color='steelblue', edgecolor='white')
axes[0,0].set_title('Distribution of Duration')
axes[0,0].set_xlabel('Duration (mins)')
axes[0,0].set_ylabel('Count')

for kind, color in zip(['short','long'], ['orange','steelblue']):
    subset = df[df['kind'] == kind]
    axes[0,1].scatter(subset['duration'], subset['waiting'],
                      label=kind, alpha=0.6, color=color)
axes[0,1].set_title('Duration vs Waiting Time')
axes[0,1].set_xlabel('Duration (mins)')
axes[0,1].set_ylabel('Waiting Time (mins)')
axes[0,1].legend()

sns.boxplot(data=df, x='kind', y='waiting', palette='Set2', ax=axes[1,0])
axes[1,0].axhline(df['waiting'].mean(), color='red', linestyle='--')
axes[1,0].set_title('Waiting Time by Type')
axes[1,0].set_xlabel('Eruption Type')
axes[1,0].set_ylabel('Waiting Time (mins)')

group_stats['mean_duration'].plot(kind='bar', ax=axes[1,1],
                                   color=['orange','steelblue'],
                                   edgecolor='white')
axes[1,1].set_title('Mean Duration by Type')
axes[1,1].set_xlabel('Eruption Type')
axes[1,1].set_ylabel('Mean Duration (mins)')
axes[1,1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('geyser_dashboard.png', dpi=150)
plt.show()
print("Dashboard saved! ✅")


# ============================================================
# CELL 10 - BONUS: Interactive Plotly chart
# ============================================================
fig = px.scatter(df,
                 x='duration',
                 y='waiting',
                 color='kind',
                 title='Interactive: Duration vs Waiting Time',
                 labels={'duration': 'Duration (mins)',
                         'waiting': 'Waiting Time (mins)'},
                 hover_data=['duration_bucket'],
                 color_discrete_map={'short': '#E84855', 'long': '#2E86AB'})
fig.show()