import pandas as pd

# Load the CSV file
df = pd.read_csv('primacare_details - tableExport (10).csv')

# Convert the date column to datetime format
df['Received On'] = pd.to_datetime(df['Received On'], errors='coerce')

# Define the cutoff date
cutoff_date = pd.Timestamp('2025-05-01')

# Filter out rows on or before May 24th
filtered_df = df[df['Received On'] > cutoff_date]

# Sort by patient name alphabetically (replace 'Patient Name' with actual column name)
filtered_df = filtered_df.sort_values(by='Patient', ascending=True)

# Save the resultprimacare_details - tableExport (10)primacare_details - tableExport (10)
filtered_df.to_csv('filtered_file.csv', index=False)

print("Rows on or before May 24th have been removed and data has been sorted by patient name.")