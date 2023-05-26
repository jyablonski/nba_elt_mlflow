CREATE SCHEMA ml_models;
SET search_path TO ml_models;

-- boostrap script boiiiiiiiii

-- table of games with today's date, built from dbt
DROP TABLE IF EXISTS ml_tonights_games;
CREATE TABLE IF NOT EXISTS ml_tonights_games
(
    home_team text COLLATE pg_catalog."default",
    away_team text COLLATE pg_catalog."default",
    home_moneyline numeric,
    away_moneyline numeric,
    proper_date date,
    home_team_rank bigint,
    home_days_rest integer,
    home_team_avg_pts_scored numeric,
    home_team_avg_pts_scored_opp numeric,
    home_team_win_pct numeric,
    home_team_win_pct_last10 numeric,
    home_is_top_players numeric,
    away_team_rank bigint,
    away_days_rest integer,
    away_team_avg_pts_scored numeric,
    away_team_avg_pts_scored_opp numeric,
    away_team_win_pct numeric,
    away_team_win_pct_last10 numeric,
    away_is_top_players numeric,
    outcome integer
);

INSERT INTO ml_tonights_games (home_team, away_team, home_moneyline, away_moneyline, proper_date, home_team_rank,
                               home_days_rest, home_team_avg_pts_scored, home_team_avg_pts_scored_opp, home_team_win_pct,
                               home_team_win_pct_last10, home_is_top_players, away_team_rank, away_days_rest,
                               away_team_avg_pts_scored, away_team_avg_pts_scored_opp, away_team_win_pct,
                               away_team_win_pct_last10, away_is_top_players, outcome)
VALUES ('Boston Celtics', 'Chicago Bulls', '-160', '200', current_date, 8, 3, 116.3, 112.4, 0.75, 0.63, 2, 14, 0, 115.3, 117.2, 0.45, 0.48, 2, null),
       ('Golden State Warriors', 'New Orleans Pelicans', '-250', '410', current_date, 4, 1, 118.3, 115.1, 0.66, 0.75, 2, 14, 2, 114.3, 112.2, 0.56, 0.44, 1, null);

-- table that the ml pipeline writes predictions to 
DROP TABLE IF EXISTS tonights_games_ml;
CREATE TABLE IF NOT EXISTS tonights_games_ml
(
    index bigint,
    home_team text COLLATE pg_catalog."default",
    home_moneyline double precision,
    away_team text COLLATE pg_catalog."default",
    away_moneyline double precision,
    proper_date text COLLATE pg_catalog."default",
    home_team_rank bigint,
    home_days_rest bigint,
    home_team_avg_pts_scored double precision,
    home_team_avg_pts_scored_opp double precision,
    home_team_win_pct double precision,
    home_team_win_pct_last10 double precision,
    home_is_top_players bigint,
    away_team_rank bigint,
    away_days_rest bigint,
    away_team_avg_pts_scored double precision,
    away_team_avg_pts_scored_opp double precision,
    away_team_win_pct double precision,
    away_team_win_pct_last10 double precision,
    away_is_top_players bigint,
    home_team_predicted_win_pct double precision,
    away_team_predicted_win_pct double precision
);