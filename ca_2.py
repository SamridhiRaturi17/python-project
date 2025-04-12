import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\DELL\Downloads\COVID-19_Daily_Rolling_Average_Case__Death__and_Hospitalization_Rates_-_Historical.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')  
# ============================  Dataset Basic Overview ============================

print("=== Basic Dataset Info ===")
print("Shape of the matrix:", df.shape)
print("\n=== Column Names ===")
print(df.columns)
print("\n=== Data Types ===")
print(df.dtypes)
print("\n=== Missing Values ===")
print(df.isnull().sum())

# ============================  EDA - Descriptive Stats for Total Case/Death/Hospitalization Rates ============================
print("\n=== Cases Rate - Total ===")
print("Mean:", df["Cases Rate - Total"].mean())
print("Median:", df["Cases Rate - Total"].median())
print("Mode:", df["Cases Rate - Total"].mode()[0])
print("Count:", df["Cases Rate - Total"].count())
print("Max:", df["Cases Rate - Total"].max())
print("Min:", df["Cases Rate - Total"].min())

print("\n=== Deaths Rate - Total ===")
print("Mean:", df["Deaths Rate - Total"].mean())
print("Median:", df["Deaths Rate - Total"].median())
print("Mode:", df["Deaths Rate - Total"].mode()[0])
print("Count:", df["Deaths Rate - Total"].count())
print("Max:", df["Deaths Rate - Total"].max())
print("Min:", df["Deaths Rate - Total"].min())

print("\n=== Hospitalizations Rate - Total ===")
print("Mean:", df["Hospitalizations Rate - Total"].mean())
print("Median:", df["Hospitalizations Rate - Total"].median())
print("Mode:", df["Hospitalizations Rate - Total"].mode()[0])
print("Count:", df["Hospitalizations Rate - Total"].count())
print("Max:", df["Hospitalizations Rate - Total"].max())
print("Min:", df["Hospitalizations Rate - Total"].min())

# ============================  Line Plot - COVID Trends Over Time ============================
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Cases Rate - Total'], label='Cases Rate', color='blue')
plt.plot(df['Date'], df['Deaths Rate - Total'], label='Deaths Rate', color='red')
plt.plot(df['Date'], df['Hospitalizations Rate - Total'], label='Hospitalizations Rate', color='green')
plt.title('COVID-19 Daily Rolling Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Rate per 100K')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================ Objective 1: Monthly Average Cases Rate (Bar Chart) ============================
print("\n📊 Objective 1: Monthly Average COVID-19 Case Rate (Bar Chart)")
df['Month'] = df['Date'].dt.strftime('%b')
monthly_cases = df.groupby('Month')[["Cases Rate - Total"]].mean().reindex([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Month', y='Cases Rate - Total', data=monthly_cases, hue='Month')
plt.title('Monthly Average COVID-19 Case Rate')
plt.xlabel('Month')
plt.ylabel('Average Case Rate')
plt.tight_layout()
plt.show()

# ============================ Objective 2: Age-wise Comparison (Pie Chart) ============================
print("\n🥧 Objective 2: Age-Based COVID-19 Case Rate Comparison (Pie Chart)")


avg_18_29 = df["Cases Rate - Age 18-29"].mean()
avg_60_plus = df["Cases Rate - Age 60-69"].mean()

labels = ['Age 18-29 (Younger Adults)', 'Age 60+ (Older Adults)']
sizes = [avg_18_29, avg_60_plus]
colors = ['#ff9999', '#66b3ff']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, startangle=140, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.title('Proportion of Average Case Rates: Age 18-29 vs Age 60+')
plt.tight_layout()
plt.show()

# ============================ Objective 3: Race-Wise Distribution (Donut Chart) ============================
print("\n🍩 Objective 3: Race-Wise COVID-19 Case Rate Distribution (Donut Chart)")

race_cols = [
    'Cases Rate - Latinx', 'Cases Rate - Asian Non-Latinx',
    'Cases Rate - Black Non-Latinx', 'Cases Rate - White Non-Latinx',
    'Cases Rate - Other Race Non-Latinx'
]
race_means = [df[col].mean() for col in race_cols]

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(race_means, labels=race_cols, autopct='%1.1f%%', startangle=140)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')  # donut effect
plt.gca().add_artist(centre_circle)
plt.title('COVID-19 Case Rate Distribution by Race')
plt.tight_layout()
plt.show()

# ============================ Objective 4: Hospitalization vs Death (Scatter Plot) ============================
print("\n🔬 Objective 4: Hospitalization Rate vs. Death Rate (Scatter Plot)")
plt.figure(figsize=(10, 6))
plt.scatter(df["Hospitalizations Rate - Total"], df["Deaths Rate - Total"], color='teal', alpha=0.6)
plt.title("Hospitalizations vs. Death Rates")
plt.xlabel("Hospitalizations Rate - Total")
plt.ylabel("Deaths Rate - Total")
plt.tight_layout()
plt.show()

# ============================ Objective 5: Distribution of Case Rates (Histogram) ============================
print("\n📈 Objective 5: Distribution of COVID-19 Case Rates (Histogram)")
plt.figure(figsize=(10, 6))
plt.hist(df["Cases Rate - Total"], bins=20, color='orange', edgecolor='black')
plt.title("Distribution of COVID-19 Case Rates")
plt.xlabel("Case Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# ============================ Final Step: Analysis Complete ============================
print("\n✅ All analysis steps are successfully completed.")
