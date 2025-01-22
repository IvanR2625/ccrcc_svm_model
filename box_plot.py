import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Code creates boxplot of means of Stages of Tissue (Stage 1-4, Normal)

# Load the Excel file
file_path = 'KIRC_Project_Data_ALL.xlsx'  # Update with the correct file path
xls = pd.ExcelFile(file_path)

# Load the 'hsa-let-7b' sheet
let_7b_data = pd.read_excel(xls, sheet_name='hsa-let-7b')

# Set the style for better visuals
sns.set(style="whitegrid")

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=let_7b_data, x='Stage', y='read_per_million_miRNAMapped', palette="Set2", legend=False)

# Add labels and title
plt.title('RPM Distribution of let-7b Across Stages', fontsize=14)
plt.xlabel('Stage', fontsize=12)
plt.ylabel('RPM (read per million miRNA mapped)', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
