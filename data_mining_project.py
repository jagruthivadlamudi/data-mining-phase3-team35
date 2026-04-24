import pandas as pd
import numpy as np
import os

os.makedirs("output", exist_ok=True)

# Load datasets
salary_df = pd.read_csv("data/Salaries.csv")
ethnicity_df = pd.read_csv("data/google_ethnicity.csv")

# Clean column names
salary_df.columns = salary_df.columns.str.strip()
ethnicity_df.columns = ethnicity_df.columns.str.strip()

print("Ethnicity columns found:")
print(ethnicity_df.columns.tolist())


# 1. SALARY DATA CLEANING


clean_salary = salary_df.copy()

clean_salary["raw_rank"] = clean_salary["rank"]
clean_salary["raw_discipline"] = clean_salary["discipline"]
clean_salary["raw_gender"] = clean_salary["gender"]
clean_salary["raw_salary"] = clean_salary["salary"]

clean_salary["rank"] = clean_salary["rank"].astype(str).str.strip().str.title()
clean_salary["discipline"] = clean_salary["discipline"].astype(str).str.strip().str.upper()
clean_salary["gender"] = clean_salary["gender"].astype(str).str.strip().str.title()
clean_salary["salary"] = pd.to_numeric(clean_salary["salary"], errors="coerce")

clean_salary["flag_missing_salary"] = clean_salary["salary"].isna()
clean_salary["flag_negative_salary"] = clean_salary["salary"] < 0
clean_salary["flag_salary_outlier"] = clean_salary["salary"] > clean_salary["salary"].quantile(0.99)
clean_salary["flag_missing_gender"] = clean_salary["gender"].isna()
clean_salary["flag_invalid_gender"] = ~clean_salary["gender"].isin(["Male", "Female"])

flag_columns = [
    "flag_missing_salary",
    "flag_negative_salary",
    "flag_salary_outlier",
    "flag_missing_gender",
    "flag_invalid_gender"
]

clean_salary["has_audit_issue"] = clean_salary[flag_columns].any(axis=1)


# 2. SALARY METRICS


total_records = len(clean_salary)
flagged_records = clean_salary["has_audit_issue"].sum()

audit_flag_density = flagged_records / total_records
data_consistency_score = 1 - audit_flag_density

grouped = clean_salary.groupby(["rank", "discipline", "gender"])["salary"].median().reset_index()

pivot_gap = grouped.pivot_table(
    index=["rank", "discipline"],
    columns="gender",
    values="salary"
).reset_index()

if "Male" in pivot_gap.columns and "Female" in pivot_gap.columns:
    pivot_gap["Adjusted_Pay_Gap"] = pivot_gap["Male"] - pivot_gap["Female"]
else:
    pivot_gap["Adjusted_Pay_Gap"] = np.nan


# 3. ETHNICITY DATA CLEANING


clean_ethnicity = ethnicity_df.copy()

if "Characteristic" in clean_ethnicity.columns:
    clean_ethnicity["raw_characteristic"] = clean_ethnicity["Characteristic"]

# Automatically find percentage columns except Characteristic/year column
percentage_cols = [col for col in clean_ethnicity.columns if col != "Characteristic"]

for col in percentage_cols:
    clean_ethnicity[f"raw_{col}"] = clean_ethnicity[col]
    clean_ethnicity[col] = (
        clean_ethnicity[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    clean_ethnicity[col] = pd.to_numeric(clean_ethnicity[col], errors="coerce")

clean_ethnicity["flag_missing_percentage"] = clean_ethnicity[percentage_cols].isna().any(axis=1)


# 4. SUMMARY METRICS


average_adjusted_pay_gap = pivot_gap["Adjusted_Pay_Gap"].mean()

summary = {
    "Total Salary Records": total_records,
    "Flagged Salary Records": int(flagged_records),
    "Audit Flag Density": round(audit_flag_density, 4),
    "Data Consistency Score": round(data_consistency_score, 4),
    "Average Adjusted Pay Gap": round(average_adjusted_pay_gap, 2)
}

summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])


# 5. SAVE OUTPUTS


clean_salary.to_csv("output/cleaned_salary_data.csv", index=False)
clean_ethnicity.to_csv("output/cleaned_google_ethnicity.csv", index=False)
pivot_gap.to_csv("output/adjusted_pay_gap.csv", index=False)
summary_df.to_csv("output/project_summary_metrics.csv", index=False)


# 6. PRINT RESULTS


print("\nPROJECT SUMMARY METRICS")
print(summary_df)

print("\nADJUSTED PAY GAP TABLE")
print(pivot_gap)

print("\nFiles saved in output folder:")
print("- cleaned_salary_data.csv")
print("- cleaned_google_ethnicity.csv")
print("- adjusted_pay_gap.csv")
print("- project_summary_metrics.csv")


# 7. VISUALIZATIONS


import matplotlib.pyplot as plt

# Graph 1: Average Salary by Gender
gender_salary = clean_salary.groupby("gender")["salary"].mean()

plt.figure(figsize=(7, 5))
gender_salary.plot(kind="bar")
plt.title("Average Salary by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Salary")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/average_salary_by_gender.png")
plt.close()

# Graph 2: Adjusted Pay Gap by Rank and Discipline
pivot_gap["Group"] = pivot_gap["rank"] + " - " + pivot_gap["discipline"]

plt.figure(figsize=(10, 6))
plt.bar(pivot_gap["Group"], pivot_gap["Adjusted_Pay_Gap"])
plt.title("Adjusted Pay Gap by Rank and Discipline")
plt.xlabel("Rank and Discipline")
plt.ylabel("Male Median Salary - Female Median Salary")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output/adjusted_pay_gap_by_group.png")
plt.close()

# Graph 3: Ethnicity Trend Over Time
year_col = "Characteristic"

plt.figure(figsize=(10,6))

clean_ethnicity[year_col] = pd.to_numeric(clean_ethnicity[year_col], errors="coerce")

plot_cols = [
    "Asian+",
    "Black+",
    "Latinx+",
    "Native American+*",
    "White+"
]

for col in plot_cols:
    plt.plot(clean_ethnicity[year_col], clean_ethnicity[col], marker='o', label=col)

plt.title("Google Workforce Ethnicity Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("output/google_ethnicity_trend.png")
plt.show()

# Graph 4: Audit Issue Summary
audit_counts = clean_salary[flag_columns].sum()

plt.figure(figsize=(10, 6))
audit_counts.plot(kind="bar")
plt.title("Audit Flag Summary")
plt.xlabel("Audit Flag Type")
plt.ylabel("Number of Records Flagged")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output/audit_flag_summary.png")
plt.close()

print("\nGraphs saved in output folder:")
print("- average_salary_by_gender.png")
print("- adjusted_pay_gap_by_group.png")
print("- google_ethnicity_trend.png")
print("- audit_flag_summary.png")
