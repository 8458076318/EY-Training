import numpy as np

print('NumPy version:', np._version_)

a = np.array([1, 2, 3, 4, 5])

print('Array:', a)
print('Shape:', a.shape)    # (5,) — 1-D, 5 elements
print('Dtype:', a.dtype)    # int64 by default
print('Ndim:', a.ndim)      # 1

mat = np.array([[1, 2, 3],
                [4, 5, 6]])

print('Shape:', mat.shape)   # (2, 3) — 2 rows, 3 cols
print(mat)

# Create 3x3 array random floats between 0 and 1.
# Print its shape, dtype, and the value at row 1, col 2.


arr = np.random.rand(3, 3)

print(arr)

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)

print("Value at row 1, col 2:", arr[1, 2])

# 1. How many unique values does the 'embarked' column have?

print("Unique values in embarked column:")
print(df['embarked'].nunique())

# To see the actual values
print(df['embarked'].unique())


# 2. What is the most common passenger class (pclass)?

print("\nMost common passenger class:")
print(df['pclass'].mode()[0])
