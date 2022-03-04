### ML FLOW
# Basics
run `src/main.py` to run the model
  
run `mlflow ui` and go to `http://localhost:5000/#/` to view historical runs.  it's just a flask server.

`mlflow run sklearn_elasticnet_wine -P alpha=0.5`
`mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0`


have the tonights ml game have the odds data and have it look exactly like it does in prod, and then in 
shiny make a selector to either select 1) tonight's games, or 2) the full remaining schedule.