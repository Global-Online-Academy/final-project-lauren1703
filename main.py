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
print(coefficients)
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

source = ColumnDataSource(coeff_df)
p = figure(x_range=coeff_df['factors'], title="Regression Coefficients", toolbar_location=None, tools="")
p.vbar(x='factors', top='weights', width=0.9, source=source, legend_field="factors")
p.xgrid.grid_line_color = None
p.y_range.start = min(coeff_df['weights']) - 1
p.y_range.end = max(coeff_df['weights']) + 1
hover2 = HoverTool()
hover2.tooltips = [
    ("Factor", "@labels"),
    ("Value", "@weights")
]
p.add_tools(hover2)
p.legend.visible = False

show(p)

