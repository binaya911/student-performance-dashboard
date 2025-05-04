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

# Simulated users (for login testing)
VALID_USERS = {'admin': 'admin123', 'student': 'student123'}

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Create login status store to track authentication state
login_status = dcc.Store(id='login-status', data={'logged_in': False})

# Load or generate data
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

# Get unique student IDs
student_ids = sorted(df['Student_ID'].unique())

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

# Login layout
login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Login to Student Dashboard", className="text-center my-4"),
            dbc.Form([
                dbc.Row([
                    dbc.Label("Username"),
                    dbc.Input(id='username-input', type='text', placeholder="Enter username"),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Password"),
                    dbc.Input(id='password-input', type='password', placeholder="Enter password"),
                ], className="mb-3"),
                html.Div(id='login-message', className='text-danger my-2'),
                dbc.Button("Login", id='login-button', color="primary", className="mt-2"),
            ])
        ], width=6)
    ], justify="center")
], className="mt-5")

# Dashboard layout
dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Student Performance Dashboard", className="text-center my-4"),
            html.Div([
                dbc.Alert(
                    f"Using {'real data from CSV file' if data_source == 'real' else 'synthetic generated data'}", 
                    color="info" if data_source == 'real' else "warning", 
                    className="text-center"
                ),
                dbc.Button("Logout", id="logout-button", color="danger", className="float-end")
            ])
        ])
    ]),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Student", className="fw-bold"),
                    dcc.Dropdown(
                        id='student-dropdown',
                        options=[{'label': f'Student {sid}', 'value': sid} for sid in student_ids],
                        value=student_ids[0] if student_ids else None,
                        clearable=False
                    ),
                ], width=4),

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

            dbc.Row([
                dbc.Col([
                    html.Button("Predict", id="predict-button", className="btn btn-primary mt-3"),
                    html.Div(id='prediction-output', className="mt-3 text-center fw-bold")
                ], width=12, className="text-center")
            ])
        ])
    ], className="mb-4"),

    # NEW: Added row for Pass/Fail and Parental Education charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Pass/Fail Distribution"),
                dbc.CardBody([
                    dcc.Graph(id='pass-fail-pie')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Exam Scores by Parental Education"),
                dbc.CardBody([
                    dcc.Graph(id='parental-education-box')
                ])
            ])
        ], width=6)
    ], className="mb-4"),

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

    dbc.Row([
        dbc.Col([
            html.H2("Courses Overview", className="text-center mb-3"),
            html.Div(id='courses-cards')
        ])
    ])
], fluid=True)

# Main layout - switches between login and dashboard
app.layout = html.Div([
    login_status,
    html.Div(id='page-content')
])

# Callback to display login or dashboard based on authentication status
@app.callback(
    Output('page-content', 'children'),
    Input('login-status', 'data')
)
def display_page(data):
    if data and data.get('logged_in'):
        return dashboard_layout
    else:
        return login_layout

# Authentication callback
@app.callback(
    Output('login-status', 'data'),
    Output('login-message', 'children'),
    Input('login-button', 'n_clicks'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    prevent_initial_call=True
)
def authenticate_user(n_clicks, username, password):
    if n_clicks is None:
        return {'logged_in': False}, ""
    
    if username in VALID_USERS and VALID_USERS[username] == password:
        return {'logged_in': True}, ""
    return {'logged_in': False}, "❌ Invalid username or password"

# Logout callback
@app.callback(
    Output('login-status', 'data', allow_duplicate=True),
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout_user(n_clicks):
    return {'logged_in': False}

# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('study-hours-slider', 'value'),
    State('attendance-slider', 'value'),
    State('student-dropdown', 'value'),
    prevent_initial_call=True
)
def predict(n_clicks, study_hours, attendance, student_id):
    if not n_clicks or not student_id:  # Check if button was clicked and student is selected
        return ""
        
    try:
        prediction = model.predict([[study_hours, attendance]])[0]
        prob = model.predict_proba([[study_hours, attendance]])[0][1]
        return html.Div([
            html.Span(f"Prediction for Student {student_id}: ", className="me-2"),
            dbc.Badge("PASS" if prediction == 1 else "FAIL", 
                      color="success" if prediction == 1 else "danger", 
                      className="p-2"),
            html.Span(f" (Confidence: {prob:.2%})" if prediction == 1 else f" (Confidence: {1-prob:.2%})")
        ])
    except Exception as e:
        return dbc.Alert(f"Error making prediction: {str(e)}", color="danger")

# Scatter plot callback
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('student-dropdown', 'value')
)
def update_scatter_plot(selected_student):
    if not selected_student:  # Check if student is selected
        return px.scatter(title="Please select a student")
        
    try:
        student_df = df[df['Student_ID'] == selected_student]
        
        if student_df.empty:
            return px.scatter(title=f"No data found for Student {selected_student}")
            
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
        return px.scatter(title=f"Error: {str(e)}")

# NEW: Callback for Pass/Fail Pie Chart and Parental Education Box Plot
@app.callback(
    [Output('pass-fail-pie', 'figure'),
     Output('parental-education-box', 'figure')],
    [Input('student-dropdown', 'value')]
)
def update_overview_charts(selected_student):
    # Pass/Fail Pie Chart (shows overall data, not student-specific)
    pass_fail_pie = px.pie(
        df,
        names='Pass_Fail',
        title='Pass/Fail Distribution',
        color='Pass_Fail',
        color_discrete_map={'Pass': '#4CAF50', 'Fail': '#F44336'},
        hole=0.3
    )
    pass_fail_pie.update_layout(legend_title_text='Result')

    # Parental Education Box Plot (shows overall data)
    parental_education_box = px.box(
        df,
        x='Parental_Education_Level',
        y='Final_Exam_Score',
        title='Exam Scores by Parental Education Level',
        color='Parental_Education_Level',
        labels={
            'Parental_Education_Level': 'Parental Education',
            'Final_Exam_Score': 'Exam Score'
        }
    )
    parental_education_box.update_layout(
        xaxis_title='Parental Education Level',
        yaxis_title='Final Exam Score',
        showlegend=False
    )
    parental_education_box.update_xaxes(tickangle=45)

    return pass_fail_pie, parental_education_box

# Dashboard components callback
@app.callback(
    [Output('performance-graph', 'figure'),
     Output('attendance-graph', 'figure'),
     Output('courses-cards', 'children')],
    [Input('student-dropdown', 'value')]
)
def update_student_data(selected_student):
    if not selected_student:  # Check if student is selected
        empty_fig = px.bar(title="Please select a student")
        empty_card = dbc.Card(
            dbc.CardBody([
                html.H5("No student selected", className="text-warning"),
                html.P("Please select a student from the dropdown menu above.")
            ])
        )
        return empty_fig, empty_fig, dbc.Row([dbc.Col(empty_card)])
    
    try:
        student_data = df[df['Student_ID'] == selected_student]

        if student_data.empty:
            empty_fig = px.bar(title=f"No data found for Student {selected_student}")
            error_card = dbc.Card(
                dbc.CardBody([
                    html.H5("No Data", className="text-warning"),
                    html.P(f"No data found for Student {selected_student}")
                ])
            )
            return empty_fig, empty_fig, dbc.Row([dbc.Col(error_card)])

        # Create subject performance graph
        performance_df = student_data.groupby('Subject')['Final_Exam_Score'].mean().reset_index()
        if performance_df.empty:
            performance_fig = px.bar(title="No subject performance data available")
        else:
            performance_fig = px.bar(
                performance_df,
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

        # Create weekly attendance and study hours graph
        weekly_data = student_data.groupby('Week').agg({
            'Attendance_Rate': 'mean',
            'Study_Hours_per_Week': 'mean'
        }).reset_index()
        
        if weekly_data.empty:
            attendance_fig = px.line(title="No weekly data available")
        else:
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

        # Create subject summary cards
        subject_summary = student_data.groupby('Subject').agg({
            'Attendance_Rate': 'mean',
            'Final_Exam_Score': 'mean',
            'Study_Hours_per_Week': 'mean'
        }).reset_index()

        if subject_summary.empty:
            error_card = dbc.Card(
                dbc.CardBody([
                    html.H5("No Subject Data", className="text-warning"),
                    html.P("No subject data available for this student.")
                ])
            )
            cards = dbc.Row([dbc.Col(error_card)])
        else:
            cards = []
            for _, row in subject_summary.iterrows():
                grade = get_grade(row['Final_Exam_Score'])
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5(row['Subject'], className="card-title")),
                            dbc.CardBody([
                                html.Div([html.Span("Attendance: ", className="fw-bold"), f"{row['Attendance_Rate']:.1f}%"], className="mb-2"),
                                html.Div([html.Span("Study Hours: ", className="fw-bold"), f"{row['Study_Hours_per_Week']:.1f} hrs/week"], className="mb-2"),
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
            cards = dbc.Row(cards)

        return performance_fig, attendance_fig, cards

    except Exception as e:
        error_message = str(e)
        empty_fig = px.bar(title=f"Error: {error_message}")
        error_card = dbc.Card(
            dbc.CardBody([
                html.H5("Error", className="text-danger"),
                html.P(f"Failed to load student data: {error_message}")
            ])
        )
        return empty_fig, empty_fig, dbc.Row([dbc.Col(error_card)])

if __name__ == '__main__':
    app.run(debug=True)