import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Initialize the Dash app with proper bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                suppress_callback_exceptions=True)

# Data loading function with proper error handling
def load_data():
    data_path = 'data/student_performance_dataset.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        data_source = 'real'
    else:
        print(f"Warning: Dataset file not found at {data_path}. Using synthetic data instead.")
        # Generate synthetic data
        np.random.seed(42)
        student_ids = [f'S{i:03d}' for i in range(1, 101)]
        subjects = ['Math', 'Science', 'English', 'History']
        data = []
        for student_id in student_ids:
            base_hours = np.random.randint(5, 35)
            base_attendance = np.random.randint(60, 100)
            for subject in subjects:
                for week in range(1, 17):
                    study_hours = max(0, base_hours + np.random.randint(-5, 6))
                    attendance = min(100, max(0, base_attendance + np.random.randint(-10, 11)))
                    exam_score = min(100, max(0, attendance * 0.3 + study_hours * 2 + np.random.randint(-20, 21)))
                    pass_fail = 1 if exam_score >= 60 else 0
                    sleep_hours = np.random.randint(4, 10)
                    data.append({
                        'Student_ID': student_id,
                        'Subject': subject,
                        'Week': week,
                        'Study_Hours_per_Week': study_hours,
                        'Attendance_Rate': attendance,
                        'Final_Exam_Score': exam_score,
                        'Pass_Fail': pass_fail,
                        'Sleep_Hours': sleep_hours
                    })
        df = pd.DataFrame(data)
        data_source = 'synthetic'
    
    # Data preprocessing
    df.fillna(df.median(numeric_only=True), inplace=True)
    df = df.dropna()
    
    return df, data_source

# Load data
df, data_source = load_data()

# Train ML model
def get_trained_model():
    X = df[['Study_Hours_per_Week', 'Attendance_Rate']]
    y = df['Pass_Fail']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Create the model once at startup
model = get_trained_model()

# Utility functions
def get_grade(score):
    if score >= 85:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 60:
        return 'C'
    else:
        return 'D'

def get_grade_color(grade):
    if grade == 'A': return 'success'
    elif grade == 'B': return 'primary'
    elif grade == 'C': return 'warning'
    else: return 'danger'

# Get unique student IDs
student_ids = sorted(df['Student_ID'].unique())

# App layout with improved UI
app.layout = html.Div([
    dbc.Container([
        # Header with data source notification
        dbc.Row([
            dbc.Col([
                html.H1("Student Performance Dashboard", className="text-center my-4"),
                html.Div(
                    dbc.Alert(
                        f"Using {'real data from CSV file' if data_source == 'real' else 'synthetic generated data'}", 
                        color="info" if data_source == 'real' else "warning", 
                        className="text-center"
                    ),
                )
            ])
        ]),
        
        # Controls in a card for better organization
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Student selection
                    dbc.Col([
                        html.Label("Select Student", className="fw-bold"),
                        dcc.Dropdown(
                            id='student-dropdown',
                            options=[{'label': f'Student {sid}', 'value': sid} for sid in student_ids],
                            value=student_ids[0],
                            clearable=False
                        ),
                    ], width=4),
                    
                    # Prediction controls
                    dbc.Col([
                        html.H5("Performance Predictor", className="mb-3"),
                        html.Div([
                            html.Label("Study Hours per Week", className="fw-bold d-block"),
                            dcc.Slider(
                                id='study-hours-slider', 
                                min=0, 
                                max=40, 
                                step=1, 
                                value=20,
                                marks={i: str(i) for i in range(0, 41, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-3"),
                        html.Div([
                            html.Label("Attendance Rate (%)", className="fw-bold d-block"),
                            dcc.Slider(
                                id='attendance-slider', 
                                min=50, 
                                max=100, 
                                step=1, 
                                value=80,
                                marks={i: str(i) for i in range(50, 101, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ]),
                    ], width=8),
                ]),
                
                # Prediction output
                dbc.Row([
                    dbc.Col([
                        html.Button("Predict", id="predict-button", className="btn btn-primary mt-3"),
                        html.Div(id='prediction-output', className="mt-3 text-center fw-bold")
                    ], width=12, className="text-center")
                ])
            ])
        ], className="mb-4"),
        
        # Visual Graphs with better layout
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Study Hours vs Exam Score"),
                    dbc.CardBody([
                        dcc.Graph(id='scatter-plot')
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance by Subject"),
                    dbc.CardBody([
                        dcc.Graph(id='performance-graph')
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Weekly Attendance & Study Hours"),
                    dbc.CardBody([
                        dcc.Graph(id='attendance-graph')
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Course Overview
        dbc.Row([
            dbc.Col([
                html.H2("Courses Overview", className="text-center mb-3"),
                html.Div(id='courses-cards')
            ])
        ])
    ], fluid=True)
])

# Prediction callback - only update on button click for better UX
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('study-hours-slider', 'value'),
    State('attendance-slider', 'value'),
    State('student-dropdown', 'value'),
    prevent_initial_call=True
)
def predict(n_clicks, study_hours, attendance, student_id):
    if n_clicks is None:
        return ""
    
    try:
        prediction = model.predict([[study_hours, attendance]])[0]
        prob = model.predict_proba([[study_hours, attendance]])[0][1]
        
        # Return styled prediction output
        if prediction == 1:
            return html.Div([
                html.Span(f"Prediction for Student {student_id}: ", className="me-2"),
                dbc.Badge("PASS", color="success", className="p-2"),
                html.Span(f" (Confidence: {prob:.2%})")
            ])
        else:
            return html.Div([
                html.Span(f"Prediction for Student {student_id}: ", className="me-2"),
                dbc.Badge("FAIL", color="danger", className="p-2"),
                html.Span(f" (Confidence: {1-prob:.2%})")
            ])
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Error making prediction: {str(e)}", color="danger")
        ])

# Scatter plot callback
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('student-dropdown', 'value')
)
def update_scatter_plot(selected_student):
    try:
        # Create scatter plot with highlighted selected student
        student_df = df[df['Student_ID'] == selected_student]
        other_df = df[df['Student_ID'] != selected_student]
        
        fig = px.scatter(
            other_df, 
            x='Study_Hours_per_Week', 
            y='Final_Exam_Score',
            opacity=0.5,
            color='Pass_Fail',
            color_discrete_map={1: 'green', 0: 'red'},
            labels={'Pass_Fail': 'Result', 1: 'Pass', 0: 'Fail'}
        )
        
        # Add the selected student with larger size
        fig.add_scatter(
            x=student_df['Study_Hours_per_Week'],
            y=student_df['Final_Exam_Score'],
            mode='markers',
            marker=dict(size=12, color='blue', line=dict(width=2, color='black')),
            name=f'Student {selected_student}'
        )
        
        fig.update_layout(
            xaxis_title="Study Hours per Week",
            yaxis_title="Final Exam Score",
            legend_title="Result",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        return fig
    except Exception as e:
        # Return empty figure with error message
        fig = px.scatter(title=f"Error: {str(e)}")
        return fig

# Performance graph and attendance graph callbacks
@app.callback(
    [Output('performance-graph', 'figure'),
     Output('attendance-graph', 'figure'),
     Output('courses-cards', 'children')],
    [Input('student-dropdown', 'value')]
)
def update_student_data(selected_student):
    try:
        # Filter for selected student data
        student_data = df[df['Student_ID'] == selected_student]
        
        # Performance graph
        performance_fig = px.bar(
            student_data.groupby('Subject')['Final_Exam_Score'].mean().reset_index(),
            x='Subject', 
            y='Final_Exam_Score',
            color='Subject',
            text_auto=True
        )
        performance_fig.update_layout(
            xaxis_title="Subject",
            yaxis_title="Average Exam Score",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        # Attendance graph
        weekly_data = student_data.groupby('Week').agg({
            'Attendance_Rate': 'mean',
            'Study_Hours_per_Week': 'mean'
        }).reset_index()
        
        attendance_fig = px.line(
            weekly_data, 
            x='Week',
            y=['Attendance_Rate', 'Study_Hours_per_Week'],
            labels={'value': 'Value', 'variable': 'Metric'},
            color_discrete_map={
                'Attendance_Rate': 'blue',
                'Study_Hours_per_Week': 'green'
            }
        )
        attendance_fig.update_layout(
            xaxis_title="Week Number",
            legend_title="Metric",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        # Course Cards
        subject_summary = student_data.groupby('Subject').agg({
            'Attendance_Rate': 'mean',
            'Final_Exam_Score': 'mean',
            'Study_Hours_per_Week': 'mean'
        }).reset_index()
        
        cards = []
        for _, row in subject_summary.iterrows():
            grade = get_grade(row['Final_Exam_Score'])
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5(row['Subject'], className="card-title")),
                        dbc.CardBody([
                            html.Div([
                                html.Span("Attendance: ", className="fw-bold"),
                                f"{row['Attendance_Rate']:.1f}%"
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Study Hours: ", className="fw-bold"),
                                f"{row['Study_Hours_per_Week']:.1f} hrs/week"
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Exam Score: ", className="fw-bold"),
                                dbc.Progress(
                                    value=row['Final_Exam_Score'], 
                                    max=100, 
                                    striped=True,
                                    color=get_grade_color(grade), 
                                    className="mb-2",
                                    label=f"{row['Final_Exam_Score']:.0f}%"
                                )
                            ]),
                            html.Div([
                                html.Span("Grade: ", className="fw-bold"),
                                dbc.Badge(
                                    grade, 
                                    color=get_grade_color(grade), 
                                    className="p-2 fs-5"
                                )
                            ])
                        ])
                    ], className="h-100 shadow")
                ], width=3, className="mb-4")
            )
        
        return performance_fig, attendance_fig, dbc.Row(cards)
    
    except Exception as e:
        # Return empty figures and error message card on error
        empty_fig = px.bar(title="Error loading data")
        error_card = dbc.Card(
            dbc.CardBody([
                html.H5("Error", className="text-danger"),
                html.P(f"Failed to load student data: {str(e)}")
            ])
        )
        return empty_fig, empty_fig, dbc.Row([dbc.Col(error_card)])

# Main
if __name__ == '__main__':
    app.run(debug=True)