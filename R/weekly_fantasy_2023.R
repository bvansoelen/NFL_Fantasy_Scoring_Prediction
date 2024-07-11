rm(list=ls())
library(nflreadr)
library(dplyr)
library(slider)
library(vctrs)

nfl <- load_ff_opportunity(
  seasons = c(2021, 2022, 2023),
  stat_type = c("weekly"),
  model_version = c("latest")
) %>%
  filter(week <= 18)


team_stats <- unique(
  nfl %>%
    select(season, posteam, week, game_id, pass_yards_gained_team, rush_yards_gained_team, 
           pass_touchdown_team, rush_touchdown_team, pass_interception_team, rec_fumble_lost_team, rush_fumble_lost_team,
           pass_attempt_team
           )
) %>%
  mutate(
    opponent = mapply(function(gid, pteam) {
      gsub('_', '', gsub(pteam, "", substring(gid, 8, nchar(gid)), fixed = TRUE))
    }, game_id, posteam)
  ) %>% 
  relocate(opponent, .after = posteam) %>%
  select(-game_id)

team_stats$opponent[team_stats$opponent == 'C'] <- 'LAC'

rolling_avg_fun <- function(x) {
  if (length(x) > 1) {
    mean(x[-length(x)])
  } else {
    NA
  }
}


offensive_stats <- team_stats %>%
  group_by(posteam, season) %>%
  mutate(
    total_yards = pass_yards_gained_team + rush_yards_gained_team,
    total_tds = pass_touchdown_team + rush_fumble_lost_team,
    across(4:12,
           ~slide_dbl(.x, rolling_avg_fun, .before = Inf, .complete = FALSE),
           .names = "seasonavg_{col}"),
    across(4:12,
           ~slide_dbl(.x, rolling_avg_fun, .before = 4, .complete = TRUE),
           .names = "last4avg_{col}"),
    across(4:12,
           ~slide_dbl(.x, rolling_avg_fun, .before = 1, .complete = TRUE),
           .names = "lastweekavg_{col}")
  ) %>%
  select(season, posteam, week, seasonavg_total_yards, seasonavg_total_tds, last4avg_total_yards, last4avg_total_tds)


offense_ranked <- offensive_stats %>%
  group_by(week, season) %>%
  mutate(offense_rank = dense_rank(desc(seasonavg_total_yards)))


defensive_stats <- team_stats %>%
  group_by(opponent, season) %>%
  mutate(
    total_yards = pass_yards_gained_team + rush_yards_gained_team,
    total_tds = pass_touchdown_team + rush_fumble_lost_team,
    across(4:12,
           ~slide_dbl(.x, rolling_avg_fun, .before = Inf, .complete = FALSE),
           .names = "seasonavg_{col}"),
    across(4:12,
           ~slide_dbl(.x, rolling_avg_fun, .before = 4, .complete = TRUE),
           .names = "last4avg_{col}"),
    across(4:12,
           ~slide_dbl(.x, rolling_avg_fun, .before = 1, .complete = TRUE),
           .names = "lastweekavg_{col}")
  ) %>%
  select(season, opponent, week, seasonavg_total_yards, seasonavg_total_tds, last4avg_total_yards, last4avg_total_tds)

defense_ranked <- defensive_stats %>%
  group_by(week, season) %>%
  mutate(defense_rank = dense_rank((seasonavg_total_yards)))


# Player level
player_stats <- nfl %>%
  select(1:10, 13:14, 17:19, 23:25, 29:31, 41, 45, 46, 59:60) %>%
  mutate(fumble_lost = rec_fumble_lost + rush_fumble_lost) %>%
  select(-rec_fumble_lost, -rush_fumble_lost)


player_stats <- player_stats %>%
  filter(!is.na(player_id)) %>%
  group_by(player_id, season) %>%
  mutate(
    opponent = mapply(function(gid, pteam) {
      gsub('_', '', gsub(pteam, "", substring(gid, 8, nchar(gid)), fixed = TRUE))
    }, game_id, posteam),
    across(7:22,
            ~slide_dbl(.x, rolling_avg_fun, .before = Inf, .complete = FALSE),
            .names = "seasonavg_{col}"),
     across(7:22,
            ~slide_dbl(.x, rolling_avg_fun, .before = 4, .complete = TRUE),
            .names = "last4avg_{col}"),
     across(7:22,
           ~slide_dbl(.x, rolling_avg_fun, .before = 1, .complete = TRUE),
           .names = "lastweek_{col}"),
    home = if_else(sapply(strsplit(game_id, "_"), function(x) x[3]) == posteam, 0, 1)
    ) %>%
  relocate(opponent, .after=posteam) %>%
  relocate(home, .after=game_id)

# Issue w/ chargers being C when they play the Rams
player_stats$opponent[player_stats$opponent == 'C'] <- 'LAC'


player_w_team_ranks <- player_stats %>%
  left_join(
    defense_ranked %>% select(season, week, opponent, defense_rank), 
    by = c('week', 'opponent', 'season')
  ) %>% 
  left_join (
    offense_ranked %>% select(season, week, posteam, offense_rank),
    by = c('week', 'posteam', 'season')
  ) %>%
  relocate(defense_rank, .after = opponent) %>%
  relocate(offense_rank, .after = posteam)


# Get depth chart data
depth_chart <- load_depth_charts(c(2021, 2022, 2023)) %>%
  # filter(gsis_id %in% player_w_defense_rank$player_id) %>%
  mutate(season = as.character(season)) %>%
  filter(formation == 'Offense') %>%
  select(week, gsis_id, season, depth_team) %>%
  distinct()


# player_final <- player_w_injury %>%
#   left_join(depth_chart, join_by(week == week, player_id == gsis_id))

player_final <- player_w_team_ranks %>%
  left_join(depth_chart, 
            join_by(week == week, player_id == gsis_id, season == season)
            )


## Get player injuries
injuries <- load_injuries(c(2021, 2022, 2023)) %>%
  filter(week <= 18) %>%
         #& gsis_id %in% player_stats$player_id) %>%
  mutate(
    injury = if_else(is.na(practice_primary_injury), report_primary_injury, practice_primary_injury)
  ) %>%
  filter(position %in% c('QB', 'RB', 'WR', 'TE', 'T', 'C', 'G')) %>%
  select('season', 'week', 'team', 'gsis_id', 'injury', 'report_status', 'practice_status', 'position')


# Join injury data w/ player data
# player_w_injury <- player_w_defense_rank %>%
#   left_join(injuries, join_by(week == week, player_id == gsis_id))

out_weeks <- injuries %>% 
  filter(report_status == 'Out') %>%
  mutate(season = as.character(season)) %>%
  inner_join(depth_chart, by = c('gsis_id', 'week', 'season')) 


out_weeks_team_level <- out_weeks %>%
  group_by(season, week, team) %>%
  summarize(n_players_out = n_distinct(gsis_id),
            n_starters_out = n_distinct(if_else(depth_team == 1, gsis_id, NA), na.rm = T),
            n_second_string_out = n_distinct(if_else(depth_team == 1, gsis_id, NA), na.rm = T),
            offensive_line_out = n_distinct(if_else(position %in% c('T', 'G', 'C'), gsis_id, NA), na.rm = T),
            qb_out = n_distinct(if_else(position == 'QB' & depth_team == 1, gsis_id, NA), na.rm = T)
            )

player_final <- player_final %>%
  left_join(out_weeks_team_level, join_by(season == season, week == week, posteam == team))


# Impute NA names
na_names <- player_final %>%
  filter(is.na(full_name))

players <- nflreadr::load_players() %>%
  filter(gsis_id %in% na_names$player_id) %>%
  select(gsis_id, display_name)

player_final <- player_final %>%
  left_join(players, by = c("player_id" = "gsis_id")) %>%
  mutate(full_name = ifelse(is.na(full_name), display_name, full_name)) %>%
  select(-display_name)

duplicates <- player_final[duplicated(player_final), ] %>%
  relocate(depth_team, .before = 'player_id')


write.csv(player_final, '2021_to_2023_ff_data_w_fpt_avg_and_team_inj.csv', row.names = FALSE)


#####################################################################
na_injuries <- player_final %>%
  filter(is.na(n_players_out))


na_off_rank <- player_final %>%
  filter(is.na(offense_rank))
na_def_rank <- player_final %>%
  filter(is.na(defense_rank) & !(game_id %in% na_off_rank$game_id))

unique(player_final$opponent)

na_seasonavg <- player_final %>%
  filter(is.na(seasonavg_rec_attempt)) %>%
  group_by(season) %>%
  summarize(total = n_distinct(player_id))

sum(na_seasonavg$total)

na_depth <- player_final %>%
  filter(is.na(depth_team))

vec_count(na_depth$position)

# player_w_injury <- bind_rows(player_w_injury, out_weeks)



## Testing 
test_d <- read.csv('/Users/bennett/Documents/Python/NFL_Analytics/NFL_Analytics_w_Python/Example/data/aggregated_2015.csv')
na_wk_5 <- player_stats %>% filter(is.na(seasonavg_rush_attempt) & week >= 5)

ag <- test_d %>% filter(playerID == '00-0021547')
