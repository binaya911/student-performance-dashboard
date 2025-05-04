# Student Dashboard

An interactive dashboard for tracking and analyzing student performance metrics.

## Features

- Student performance tracking across different subjects
- Machine learning-based pass/fail predictions
- Interactive visualizations of study hours and exam scores
- Attendance rate monitoring
- Weekly performance trends

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
python student_dashboard.py
```

3. Open your web browser and navigate to `http://127.0.0.1:8050` to view the dashboard.

## Data

The dashboard uses student performance data stored in `data/student_performance_dataset.csv`. If the file is not found, the application will generate sample data automatically.

## Usage

1. Select a student from the dropdown menu to view their performance metrics
2. Use the sliders to input study hours and attendance rate for pass/fail prediction
3. Explore the interactive graphs showing:
   - Study hours vs exam scores
   - Subject-wise performance
   - Weekly attendance and study hour trends 