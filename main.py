import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import dodge

output_notebook()

data = pd.read_csv('Final Project Data - Sheet1.csv')

# Create imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')
columns_to_impute = data[['Global rank','Average SAT Score','Average ACT Score','High School GPA',
                          'Tuition costs','Average Financial Aid','Endowment size per student (in millions)','# of undergraduate majors',
                          'Undergraduate enrollment','Graduate enrollment',
                          'Graduation rate','Average starting salaries',
                          'Student-to-faculty ratio','Bibliometric rank','# of research papers published per faculty',
                          'Citations per publication','Percent of tenured faculty']]
data[columns_to_impute.columns] = imputer.fit_transform(columns_to_impute)

# One-hot encoding
data = pd.get_dummies(data, columns=['Public or private'])
data = data.drop('Public or private_public', axis=1)

data['Public or private_private'] = data['Public or private_private'].astype(int)

print("First 5 colleges in the data table:")
print(data.head())
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Global rank','Year founded','Acceptance Rate', 'Average SAT Score',
                                         'Average ACT Score','High School GPA',
                          'Tuition costs','Average Financial Aid','Endowment size per student (in millions)','# of undergraduate majors',
                          'Undergraduate enrollment','Graduate enrollment',
                          'Graduation rate','Average starting salaries',
                          'Student-to-faculty ratio','Bibliometric rank','# of research papers published per faculty',
                          'Citations per publication','Percent of tenured faculty','Public or private_private']])

X = sm.add_constant(data_scaled)  # Independent variables
y = data['National rank']  # Dependent variable

model = sm.OLS(y, X).fit()

print(model.summary())

# Actual vs Predicted Regression Graph
predictions = model.predict(X)

plot_data = pd.DataFrame({
    'Actual': y,
    'Predicted': predictions,
    'colleges': ['Princeton University', 'Harvard University', 'California Institue of Technology',
                 'Northwestern University', 'Cornell University', 'Rice University', 'University of Notre Dame',
                  'Carnegie Mellon University', 'University of California, Davis', 'University of Texas at Austin',
                'University of California, Santa Barbara', 'Rutgers University--New Brunswick', 'Ohio State University',
                 'Texas A&M University', 'Wake Forest University', 'University of Minnesota, Twin Cities',
                 'Brandeis University', 'Rensselaer Polytechnic Institute', 'Syracuse University', 'Villanova University',
                 'Colorado School of Mines', 'University of California, Riverside', 'University of Illinois--Chicago',
                 'New Jersey Institute of Technology', 'University of South Florida', 'Loyola Marymount University']
})
source = ColumnDataSource(plot_data)

p = figure(title="Actual vs Predicted College Rankings", x_axis_label='Actual Rank', y_axis_label='Predicted Rank',
           width=800, height=400)

p.circle('Actual', 'Predicted', source=source, size=10, color="navy", alpha=0.5)
min_val = min(plot_data['Actual'].min(), plot_data['Predicted'].min())
max_val = max(plot_data['Actual'].max(), plot_data['Predicted'].max())
p.line([min_val, max_val], [min_val, max_val], color="red", line_width=2, legend_label="Perfect Fit")

hover = HoverTool()
hover.tooltips=[
    ('College', '@colleges'),
    ('(Actual, Predicted)', '(@Actual, @Predicted)')
]
p.add_tools(hover)
show(p)

coefficients = model.params
coeff_df = pd.DataFrame({
    'factors': coefficients.index,
    'weights': coefficients.values,
    'labels': ['Constant','Global rank','Year founded','Acceptance Rate', 'Average SAT Score',
                'Average ACT Score','High School GPA',
                'Tuition costs','Average Financial Aid','Endowment size per student (in millions)','# of undergraduate majors',
                'Undergraduate enrollment','Graduate enrollment',
                'Graduation rate','Average starting salaries',
                'Student-to-faculty ratio','Bibliometric rank','# of research papers published per faculty',
                'Citations per publication','Percent of tenured faculty','Public or private_private'],
})

source2 = ColumnDataSource(coeff_df)
p2 = figure(x_range=coeff_df['factors'], title="Regression Coefficients (weights of each factor)", toolbar_location=None, tools="")
p2.vbar(x='factors', top='weights', width=0.9, source=source2, legend_field="factors")
p2.xgrid.grid_line_color = None
p2.y_range.start = min(coeff_df['weights']) - 1
p2.y_range.end = max(coeff_df['weights']) + 1
hover2 = HoverTool()
hover2.tooltips = [
    ("Factor", "@labels"),
    ("Value", "@weights")
]
p2.add_tools(hover2)
p2.legend.visible = False

show(p2)

results_summary = model.summary2().tables[1]

# Filter factors with p-values less than 0.05
significant_factors = results_summary[results_summary['P>|t|'] < 0.1]

labels = {"const":"Constant","x1": 'Global rank',"x2":'Year founded',"x3":'Acceptance Rate', "x4":'Average SAT Score',
          "x5":'Average ACT Score',"x6":'High School GPA',
          "x7":'Tuition costs',"x8":'Average Financial Aid',"x9":'Endowment size per student (in millions)',
          "x10":'# of undergraduate majors',"x11":'Undergraduate enrollment',"x12":'Graduate enrollment',
          "x13":'Graduation rate',"x14":'Average starting salaries',
          "x15":'Student-to-faculty ratio',"x16":'Bibliometric rank',"x17":'# of research papers published per faculty',
          "x18":'Citations per publication',"x19":'Percent of tenured faculty',"x20":'Public or private_private'}
factor_names = []
for f in significant_factors.index:
  factor_names.append(labels[f])

sig_df = pd.DataFrame({
    'factors': significant_factors.index,
    'weights': significant_factors['Coef.'],
    'labels': factor_names
})

source3 = ColumnDataSource(sig_df)
p3 = figure(x_range=sig_df['factors'], title="Statistically Significant Regression Coefficients (weights of each factor with p-val < 0.1)", toolbar_location=None, tools="")
p3.vbar(x='factors', top='weights', width=0.9, source=source3, legend_field="factors")
p3.xgrid.grid_line_color = None
p3.y_range.start = min(coeff_df['weights']) - 1
p3.y_range.end = max(coeff_df['weights']) + 1
hover3 = HoverTool()
hover3.tooltips = [
    ("Factor", "@labels"),
    ("Value", "@weights")
]
p3.add_tools(hover3)
p3.legend.visible = False

show(p3)
