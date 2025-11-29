CREATE SCHEMA silver;
CREATE SCHEMA gold;
SET search_path TO silver;

-- boostrap script boiiiiiiiii

DROP TABLE IF EXISTS ml_game_features;
CREATE TABLE IF NOT EXISTS ml_game_features
(
    home_team text COLLATE pg_catalog."default",
    away_team text COLLATE pg_catalog."default",
    home_moneyline numeric,
    away_moneyline numeric,
    game_date date,
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

INSERT INTO ml_game_features (home_team, away_team, home_moneyline, away_moneyline, game_date, home_team_rank,
                               home_days_rest, home_team_avg_pts_scored, home_team_avg_pts_scored_opp, home_team_win_pct,
                               home_team_win_pct_last10, home_is_top_players, away_team_rank, away_days_rest,
                               away_team_avg_pts_scored, away_team_avg_pts_scored_opp, away_team_win_pct,
                               away_team_win_pct_last10, away_is_top_players, outcome)
VALUES ('Boston Celtics', 'Chicago Bulls', '-160', '200', current_date, 8, 3, 116.3, 112.4, 0.75, 0.63, 2, 14, 0, 115.3, 117.2, 0.45, 0.48, 2, null),
       ('Atlanta Hawks', 'New Orleans Pelicans', '-250', '410', current_date, 4, 1, 118.3, 115.1, 0.66, 0.75, 2, 14, 2, 114.3, 112.2, 0.56, 0.44, 1, null),
       ('Denver Nuggets','Los Angeles Lakers','-200.0','165.0',current_date,4,133,115.4,111.0,0.646,0.5,2.0,14,156,116.3,115.4,0.524,0.8,2.0,NULL),
	   ('Golden State Warriors','Phoenix Suns','-160.0','130.0',current_date,12,164,118.1,116.5,0.537,0.8,2.0,10,165,113.7,112.2,0.549,0.7,2.0,NULL);

CREATE TABLE IF NOT EXISTS silver.ml_game_features_v2 (
	home_team text NULL,
	away_team text NULL,
	home_moneyline numeric NULL,
	away_moneyline numeric NULL,
	game_date date NULL,
	home_team_rank int8 NULL,
	home_days_rest numeric NULL,
	home_team_avg_pts_scored numeric NULL,
	home_team_avg_pts_scored_opp numeric NULL,
	home_team_win_pct numeric NULL,
	home_team_win_pct_last10 numeric NULL,
	home_star_score int8 NULL,
	home_active_vorp float8 NULL,
	home_pct_vorp_missing numeric NULL,
	home_travel_miles_last_7_days float8 NULL,
	home_games_last_7_days int8 NULL,
	home_is_cross_country_trip int4 NULL,
	away_team_rank int8 NULL,
	away_days_rest numeric NULL,
	away_team_avg_pts_scored numeric NULL,
	away_team_avg_pts_scored_opp numeric NULL,
	away_team_win_pct numeric NULL,
	away_team_win_pct_last10 numeric NULL,
	away_star_score int8 NULL,
	away_active_vorp float8 NULL,
	away_pct_vorp_missing numeric NULL,
	away_travel_miles_last_7_days float8 NULL,
	away_games_last_7_days int8 NULL,
	away_is_cross_country_trip int4 NULL,
	travel_miles_differential float8 NULL,
	star_score_differential int8 NULL,
	active_vorp_differential float8 NULL,
	outcome text NULL
);

INSERT INTO silver.ml_game_features_v2 (home_team,away_team,home_moneyline,away_moneyline,game_date,home_team_rank,home_days_rest,home_team_avg_pts_scored,home_team_avg_pts_scored_opp,home_team_win_pct,home_team_win_pct_last10,home_star_score,home_active_vorp,home_pct_vorp_missing,home_travel_miles_last_7_days,home_games_last_7_days,home_is_cross_country_trip,away_team_rank,away_days_rest,away_team_avg_pts_scored,away_team_avg_pts_scored_opp,away_team_win_pct,away_team_win_pct_last10,away_star_score,away_active_vorp,away_pct_vorp_missing,away_travel_miles_last_7_days,away_games_last_7_days,away_is_cross_country_trip,travel_miles_differential,star_score_differential,active_vorp_differential,outcome) VALUES
	 ('Phoenix Suns','Denver Nuggets',160,-162,'2025-11-29',21,0,116.9,113.2,0.600,0.700,1,3.0999999999999996,2.94,2812.0,5,1,20,0,124.5,115.2,0.722,0.700,3,3.8000000000000003,8.33,2343.0,3,0,469.0,-2,-0.7000000000000006,NULL),
	 ('Golden State Warriors','New Orleans Pelicans',-315,285,'2025-11-29',23,2,115.1,114.5,0.500,0.500,0,1.3,35.71,0.0,3,0,30,2,111.5,122.5,0.158,0.100,0,0.5,12.50,1920.0,3,0,-1920.0,0,0.8,NULL),
	 ('Charlotte Hornets','Toronto Raptors',320,-340,'2025-11-29',12,0,115.7,120.6,0.263,0.200,1,0.6000000000000002,0.00,454.0,4,0,2,2,119.2,112.5,0.737,0.900,1,2.9,11.43,587.0,4,0,-133.0,0,-2.3,NULL),
	 ('Minnesota Timberwolves','Boston Celtics',-255,240,'2025-11-29',22,2,118.8,114.2,0.556,0.600,2,2.8,0.00,2666.0,3,0,8,2,114.8,110.2,0.556,0.700,1,3.2,0.00,1121.0,3,0,1545.0,1,-0.40000000000000036,NULL),
	 ('Los Angeles Clippers','Dallas Mavericks',-255,245,'2025-11-29',27,0,111.9,117.9,0.263,0.200,1,1.5,8.33,2492.0,4,0,29,0,110.0,116.2,0.250,0.200,0,0.6999999999999998,7.69,3457.0,3,1,-965.0,1,0.8000000000000002,NULL),
	 ('Miami Heat','Detroit Pistons',-144,140,'2025-11-29',3,2,122.9,117.2,0.684,0.800,1,3.6000000000000005,0.00,1685.0,4,0,1,0,118.8,112.8,0.789,0.800,2,3.4,0.00,2817.0,4,0,-1132.0,-1,0.20000000000000062,NULL),
	 ('Indiana Pacers','Chicago Bulls',146,-156,'2025-11-29',14,0,110.1,119.5,0.158,0.200,0,-0.40000000000000024,0.00,1143.0,4,0,10,0,121.0,124.2,0.500,0.300,1,1.7000000000000002,0.00,1912.0,3,0,-769.0,-1,-2.1000000000000005,NULL),
	 ('Milwaukee Bucks','Brooklyn Nets',-520,495,'2025-11-29',11,0,115.5,118.6,0.400,0.200,2,1.5,0.00,3097.0,4,0,13,0,108.9,119.4,0.167,0.200,0,0.4000000000000001,0.00,1508.0,4,0,1589.0,2,1.0999999999999999,NULL);

-- table that the ml pipeline writes predictions to 
DROP TABLE IF EXISTS gold.ml_game_predictions;
CREATE TABLE IF NOT EXISTS gold.ml_game_predictions
(
    index bigint,
    home_team text COLLATE pg_catalog."default",
    home_moneyline double precision,
    away_team text COLLATE pg_catalog."default",
    away_moneyline double precision,
    game_date text COLLATE pg_catalog."default",
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

DROP TABLE IF EXISTS gold.feature_flags;
CREATE TABLE IF NOT EXISTS gold.feature_flags
(
	id serial primary key,
	flag text,
	is_enabled integer,
	created_at timestamp without time zone default now(),
	modified_at timestamp without time zone default now(),
    CONSTRAINT flag_unique UNIQUE (flag)
);
INSERT INTO gold.feature_flags(flag, is_enabled)
VALUES ('season', 1),
       ('playoffs', 0);


CREATE TABLE IF NOT EXISTS gold.ml_game_predictions_v2 (
	home_team text NULL,
	away_team text NULL,
	home_moneyline numeric NULL,
	away_moneyline numeric NULL,
	game_date date NULL,
	home_team_rank int8 NULL,
	home_days_rest numeric NULL,
	home_team_avg_pts_scored numeric NULL,
	home_team_avg_pts_scored_opp numeric NULL,
	home_team_win_pct numeric NULL,
	home_team_win_pct_last10 numeric NULL,
	home_star_score int8 NULL,
	home_active_vorp float8 NULL,
	home_pct_vorp_missing numeric NULL,
	home_travel_miles_last_7_days float8 NULL,
	home_games_last_7_days int8 NULL,
	home_is_cross_country_trip int4 NULL,
	away_team_rank int8 NULL,
	away_days_rest numeric NULL,
	away_team_avg_pts_scored numeric NULL,
	away_team_avg_pts_scored_opp numeric NULL,
	away_team_win_pct numeric NULL,
	away_team_win_pct_last10 numeric NULL,
	away_star_score int8 NULL,
	away_active_vorp float8 NULL,
	away_pct_vorp_missing numeric NULL,
	away_travel_miles_last_7_days float8 NULL,
	away_games_last_7_days int8 NULL,
	away_is_cross_country_trip int4 NULL,
	travel_miles_differential float8 NULL,
	star_score_differential int8 NULL,
	active_vorp_differential float8 NULL,
	home_team_predicted_win_pct float8 NULL,
	away_team_predicted_win_pct float8 NULL,
	created_at timestamp default current_timestamp,
	CONSTRAINT unique_constraint_for_upsert_ml_game_predictions_v2 UNIQUE (home_team, game_date)
);
