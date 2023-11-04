# ML Pipeline for NBA ELT Project
![Tests](https://github.com/jyablonski/nba_elt_mlflow/actions/workflows/test_ml.yml/badge.svg) ![Deployment](https://github.com/jyablonski/nba_elt_mlflow/actions/workflows/deploy_ml.yml/badge.svg) ![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

Version: 1.6.3

## ML Script

The ML Script pulls data generated by dbt for today's NBA Games and uses the Logistic Regression Model to create win prediction %s for every team.

## Tests
To run tests locally, run `make test`.

The same Test Suite is ran after every commit on a PR via GitHub Actions.

## Project
![NBA ELT Pipeline Data Flow](https://github.com/jyablonski/nba_elt_mlflow/assets/16946556/c14625e9-c08f-4806-aacc-a22d6f338c81)

1. Links to other Repos providing infrastructure for this Project
    * [Shiny Server](https://github.com/jyablonski/NBA-Dashboard)
    * [Ingestion Script](https://github.com/jyablonski/python_docker)
    * [dbt](https://github.com/jyablonski/nba_elt_dbt)
    * [Terraform](https://github.com/jyablonski/aws_terraform)
    * [Airflow Proof of Concept](https://github.com/jyablonski/nba_elt_airflow)
    * [REST API](https://github.com/jyablonski/nba_elt_rest_api)
