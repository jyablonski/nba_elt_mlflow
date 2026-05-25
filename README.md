# ML Pipeline for NBA ELT Project

![Workflows](https://github.com/jyablonski/nba_elt_mlflow/actions/workflows/ci_cd.yml/badge.svg)
![Coverage Status](https://coveralls.io/repos/github/jyablonski/nba_elt_mlflow/badge.svg)

## ML Pipeline

The ML Pipeline pulls input data built by dbt for today's NBA Games and uses a Logistic Regression Model to generate Win Prediction %s for every team. These Win Predictions are served on the Dash Server and the REST API.

After predictions are committed to the warehouse, this script submits a few HTTP requests to the dashboard to trigger a data refresh and monitor the health of the dashboard's data.

- This typically would be done in a separate task or script, but it's included here for simplicity since I don't have a full orchestration tool available.

The `ml_experiments` directory contains the code for training and evaluating the model, as well as for testing out other models and strategies.

## Tests

Integration tests spin up Postgres via [Testcontainers](https://testcontainers.com/). Docker must be running.

```bash
make test
```

The same test suite runs on every PR via GitHub Actions.

## Project

![nba_pipeline_diagram](https://github.com/jyablonski/nba_elt_mlflow/assets/16946556/b66284b0-147a-449c-98e4-5ac269cf5a55)

1. Links to other repos that provide infrastructure for this project
   - [Dash Server](https://github.com/jyablonski/nba_elt_dashboard)
   - [Ingestion Script](https://github.com/jyablonski/nba_elt_ingestion)
   - [dbt](https://github.com/jyablonski/nba_elt_dbt)
   - [Terraform](https://github.com/jyablonski/aws_terraform)
   - [REST API](https://github.com/jyablonski/nba_elt_rest_api)
   - [Internal Documentation](https://github.com/jyablonski/doqs)
