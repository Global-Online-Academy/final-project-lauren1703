import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.impute import SimpleImputer

output_notebook()  
data = pd.read_csv('Final Project Data - Sheet1.csv')
print(data.head())  

data.describe()

# Create imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

columns_to_impute = data[['Average SAT Score','Average ACT Score','High School GPA',
                          'Tuition costs','Average Financial Aid','Endowment size per student (in millions)','# of undergraduate majors',
                          'Undergraduate enrollment','Graduate enrollment',
                          'Graduation rate','Average starting salaries',
                          'Student-to-faculty ratio','Bibliometric rank','# of research papers published per faculty',
                          'Citations per publication','Percent of tenured faculty']]

data[columns_to_impute.columns] = imputer.fit_transform(columns_to_impute)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Acceptance Rate', 'Average SAT Score', 'Student-to-faculty ratio']])

# Adding a constant to the model (statsmodels does not add it by default)
X = sm.add_constant(data_scaled)  # Independent variables
y = data['National rank']  # Dependent variable

model = sm.OLS(y, X).fit()

print(model.summary())
