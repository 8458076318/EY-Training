import pandas as pd
import sqlite3

# Create an in-memory SQLite database connection
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

print("Setup complete. Environment ready.")

# Define a messy, unnormalized dataset (0NF)
# Issues: Non-atomic values (multiple text books), redundant data, mixed concerns.
data_0nf = {
    "Student_ID": [101, 101, 102, 103, 104],
    "Student_Name": ["Alice", "Alice", "Bob", "Charlie", "David"],
    "Course_Code": ["CS101", "MATH201", "CS101", "CS102", "MATH201"],
    "Course_Name": ["Intro to CS", "Calculus I", "Intro to CS", "Data Structures", "Calculus I"],
    "Instructor_ID": ["INS_01", "INS_02", "INS_01", "INS_03", "INS_05"],
    "Instructor_Name": ["Dr. Smith", "Dr. Jones", "Dr. Smith", "Dr. Alan", "Dr. Jones"],
    "Instructor_Office": ["Room 401", "Room 502", "Room 401", "Room 401", "Room 502"],
    "Textbooks_Required": ["Python Basics, Intro to CLI", "Calculus Vol 1", "Python Basics, Intro to CLI", "Algorithms Vol 1", "Calculus Vol 1"],
    "Grade": ["A", "B", "A", "B+", "A-"]
}

df_0nf = pd.DataFrame(data_0nf)
df_0nf.to_sql('Unnormalized_Leasing', conn, index=False, if_exists='replace')

print("\n--- Unnormalized Data (0NF) ---")
df_0nf

# ONF dummy data

# Flattening the Textbooks_Required column to ensure atomicity
df_1nf = df_0nf.assign(Textbooks_Required=df_0nf['Textbooks_Required'].str.split(', ')).explode('Textbooks_Required')

# Save to SQL
df_1nf.to_sql('Table_1NF', conn, index=False, if_exists='replace')

print("--- 1NF Data (Atomic values enforced) ---")
print(f"Row count increased from {len(df_0nf)} to {len(df_1nf)} due to flattening.")
df_1nf

# code for doing first normalization on the table

import pandas as pd
import sqlite3

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# ─────────────────────────────────────────
# ORIGINAL 0NF DATA
# ─────────────────────────────────────────
data_0nf = {
    "Student_ID": [101, 101, 102, 103, 104],
    "Student_Name": ["Alice", "Alice", "Bob", "Charlie", "David"],
    "Course_Code": ["CS101", "MATH201", "CS101", "CS102", "MATH201"],
    "Course_Name": ["Intro to CS", "Calculus I", "Intro to CS", "Data Structures", "Calculus I"],
    "Instructor_ID": ["INS_01", "INS_02", "INS_01", "INS_03", "INS_05"],
    "Instructor_Name": ["Dr. Smith", "Dr. Jones", "Dr. Smith", "Dr. Alan", "Dr. Jones"],
    "Instructor_Office": ["Room 401", "Room 502", "Room 401", "Room 401", "Room 502"],
    "Textbooks_Required": ["Python Basics, Intro to CLI", "Calculus Vol 1",
                           "Python Basics, Intro to CLI", "Algorithms Vol 1", "Calculus Vol 1"],
    "Grade": ["A", "B", "A", "B+", "A-"]
}

df_0nf = pd.DataFrame(data_0nf)
print("--- Unnormalized Data (0NF) ---")
print(df_0nf)

# ─────────────────────────────────────────
# 1NF — Atomic values + Composite PK
# ─────────────────────────────────────────
df_1nf = (df_0nf
          .assign(Textbooks_Required=df_0nf['Textbooks_Required'].str.split(', '))
          .explode('Textbooks_Required')
          .reset_index(drop=True))

df_1nf.to_sql('Table_1NF', conn, index=False, if_exists='replace')

print("\n--- 1NF Data (Atomic values enforced) ---")
print(f"Row count increased from {len(df_0nf)} to {len(df_1nf)} due to flattening.")
print(f"Primary Key: (Student_ID, Course_Code, Textbooks_Required)")
print(df_1nf)

# ─────────────────────────────────────────
# 2NF — Remove Partial Dependencies
# Split into 4 tables
# ─────────────────────────────────────────

# Table 1: Students (Student_ID → Student_Name)
df_students = df_1nf[['Student_ID', 'Student_Name']].drop_duplicates().reset_index(drop=True)
df_students.to_sql('Students', conn, index=False, if_exists='replace')

# Table 2: Courses (Course_Code → Course_Name, Instructor_ID)
df_courses = df_1nf[['Course_Code', 'Course_Name', 'Instructor_ID',
                      'Instructor_Name', 'Instructor_Office']].drop_duplicates().reset_index(drop=True)
df_courses.to_sql('Courses', conn, index=False, if_exists='replace')

# Table 3: Course_Textbooks (Course_Code + Textbooks_Required)
df_textbooks = df_1nf[['Course_Code', 'Textbooks_Required']].drop_duplicates().reset_index(drop=True)
df_textbooks.to_sql('Course_Textbooks', conn, index=False, if_exists='replace')

# Table 4: Enrollments (Student_ID + Course_Code → Grade)
df_enrollments = df_1nf[['Student_ID', 'Course_Code', 'Grade']].drop_duplicates().reset_index(drop=True)
df_enrollments.to_sql('Enrollments', conn, index=False, if_exists='replace')

# ─────────────────────────────────────────
# PRINT 2NF TABLES
# ─────────────────────────────────────────
print("\n--- 2NF Tables (Partial Dependencies Removed) ---")

print("\n[1] Students Table  |  PK: Student_ID")
print(df_students)

print("\n[2] Courses Table  |  PK: Course_Code")
print(df_courses)

print("\n[3] Course_Textbooks Table  |  PK: (Course_Code, Textbooks_Required)")
print(df_textbooks)

print("\n[4] Enrollments Table  |  PK: (Student_ID, Course_Code)")
print(df_enrollments)


# ─────────────────────────────────────────
# 2NF — Remove Partial Dependencies
# ─────────────────────────────────────────

# Students: Student_ID → Student_Name
df_students_2nf = (df_1nf[['Student_ID', 'Student_Name']]
                   .drop_duplicates().reset_index(drop=True))

# Courses (with instructor): Course_Code → Course_Name, Instructor details
df_courses_2nf = (df_1nf[['Course_Code', 'Course_Name',
                            'Instructor_ID', 'Instructor_Name', 'Instructor_Office']]
                  .drop_duplicates().reset_index(drop=True))

# Course_Textbooks: (Course_Code, Textbooks_Required)
df_textbooks_2nf = (df_1nf[['Course_Code', 'Textbooks_Required']]
                    .drop_duplicates().reset_index(drop=True))

# Enrollments: (Student_ID, Course_Code) → Grade
df_enrollments_2nf = (df_1nf[['Student_ID', 'Course_Code', 'Grade']]
                      .drop_duplicates().reset_index(drop=True))

# Save to SQL
df_students_2nf.to_sql('Students_2NF', conn, index=False, if_exists='replace')
df_courses_2nf.to_sql('Courses_2NF', conn, index=False, if_exists='replace')
df_textbooks_2nf.to_sql('Course_Textbooks_2NF', conn, index=False, if_exists='replace')
df_enrollments_2nf.to_sql('Enrollments_2NF', conn, index=False, if_exists='replace')

print("\n--- 2NF: Partial Dependencies Removed ---")
print("\n[1] Students_2NF  |  PK: Student_ID")
print(df_students_2nf)
print("\n[2] Courses_2NF  |  PK: Course_Code")
print(df_courses_2nf)
print("\n[3] Course_Textbooks_2NF  |  PK: (Course_Code, Textbooks_Required)")
print(df_textbooks_2nf)
print("\n[4] Enrollments_2NF  |  PK: (Student_ID, Course_Code)")
print(df_enrollments_2nf)

# ─────────────────────────────────────────
# 3NF — Remove Transitive Dependencies
# Transitive chain: Course_Code → Instructor_ID → Instructor_Name, Instructor_Office
# ─────────────────────────────────────────

# Extract Instructors into separate table
df_instructors_3nf = (df_courses_2nf[['Instructor_ID', 'Instructor_Name', 'Instructor_Office']]
                      .drop_duplicates().reset_index(drop=True))

# Clean up Courses — keep only Instructor_ID as Foreign Key
df_courses_3nf = (df_courses_2nf[['Course_Code', 'Course_Name', 'Instructor_ID']]
                  .drop_duplicates().reset_index(drop=True))

# Save to SQL
df_instructors_3nf.to_sql('Instructors_3NF', conn, index=False, if_exists='replace')
df_courses_3nf.to_sql('Courses_3NF', conn, index=False, if_exists='replace')

print("\n--- 3NF: Transitive Dependencies Removed ---")
print("\n[Courses_3NF]  |  PK: Course_Code  |  FK: Instructor_ID")
print(df_courses_3nf)
print("\n[Instructors_3NF]  |  PK: Instructor_ID")
print(df_instructors_3nf)