# ML Pipeline for NBA ELT Project

![Tests](https://github.com/jyablonski/nba_elt_mlflow/actions/workflows/test.yml/badge.svg) ![Deployment](https://github.com/jyablonski/nba_elt_mlflow/actions/workflows/deploy.yml/badge.svg) ![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

## ML Pipeline

The ML Pipeline pulls input data built by dbt for today's NBA Games and uses a Logistic Regression Model to generate Win Prediction %s for every team.

These Win Predictions are served on the Dash Server as well as the REST API.

## Tests

To run tests locally, run `make test`.

The same Test Suite is ran after every commit on a PR via GitHub Actions.

## Project

![nba_pipeline_diagram](https://github.com/jyablonski/nba_elt_mlflow/assets/16946556/b66284b0-147a-449c-98e4-5ac269cf5a55)

1. Links to other Repos providing infrastructure for this Project
   - [Dash Server](https://github.com/jyablonski/nba_elt_dashboard)
   - [Ingestion Script](https://github.com/jyablonski/nba_elt_ingestion)
   - [dbt](https://github.com/jyablonski/nba_elt_dbt)
   - [Terraform](https://github.com/jyablonski/aws_terraform)
   - [REST API](https://github.com/jyablonski/nba_elt_rest_api)
