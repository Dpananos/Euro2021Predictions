library(tidyverse)
library(readxl)
library(cmdstanr)
library(tidybayes)
library(posterior)
library(glue)

theme_set(theme_classic(base_size = 15))

#Load in data
uefa_nations = read_csv('data/uefa_nations_league_results.csv')


match_day_results = map_dfr(1:3, ~read_xlsx(glue('predictions/predictions_day_{.x}.xlsx')))

# COndition on match day 1 results
euro_data = read_csv('data/qualifying_round_games.csv') %>% 
            bind_rows(uefa_nations, match_day_results)

ranking_data = read_csv('data/rankings.csv') %>% 
                mutate(prior_score = (elo_march_2019 - mean(elo_march_2019))/sd(elo_march_2019)) %>% 
                arrange(team)


# extract data for model
teams = ranking_data$team
nteams = length(teams)
ngames = nrow(euro_data)
team1 = match(euro_data$team1, teams)
team2 = match(euro_data$team2, teams)
score1 = euro_data$score1
score2 = euro_data$score2
# Used for some models, not all
df = 7
b_mean = 0
b_sd = 0.05
prior_score = ranking_data$prior_score

# Store data in a list to pass to Stan
model_data = list(
  nteams = nteams,
  ngames = ngames,
  team1 = team1,
  team2 = team2,
  score1 = score1,
  score2 = score2,
  df = df,
  prior_score = prior_score,
  b_mean = b_mean,
  b_sd = b_sd
)

# Instantiate model and run sampling.
model = cmdstan_model('models/euro_raw_dif.stan')
fit = model$sample(model_data, parallel_chains=4, seed=19920908)


a = fit$draws('a') %>% as_draws_df
sigma_y = fit$draws('sigma_y')
est_df = fit$draws('df')

goal_diff = function(teamA, teamB, do_round=T){
  set.seed(0)
  ixa = match(teamA, str_to_title(teams))
  ixb = match(teamB, str_to_title(teams))
  ai = a[, ixa]
  aj = a[, ixb]
  random_outcome = (ai - aj) + rt(nrow(ai-ai), est_df)*sigma_y
  rm(.Random.seed, envir=.GlobalEnv)
  if(do_round){
    round(pull(random_outcome))
  }
  else{
    pull(random_outcome)
  }
}


predict_no_draw = function(teams){
  teamA = teams[1]
  teamB = teams[2]
  gd = goal_diff(teamA, teamB)
  #No draws in round of 16
  # This is a hack
  gd = gd[gd!=0]
  
  gdr = case_when(gd<0~-1, gd>0~1)
  
  p = mean(gdr>0)
  c(p, 1-p)
}

RO16<-function(){
  
  #Round of 16
  m1 = c('Wales','Denmark')
  m1_winner = sample(m1, size=1, prob=predict_no_draw(m1))
  
  m2 = c('Italy','Austria')
  m2_winner = sample(m2, size=1, prob=predict_no_draw(m2))
  
  m3 = c('Netherlands','Czech')
  m3_winner = sample(m3, size=1, prob=predict_no_draw(m3))
  
  m4 = c('Belgium','Portugal')
  m4_winner = sample(m4, size=1, prob=predict_no_draw(m4))
  
  m5 = c('Croatia','Spain')
  m5_winner = sample(m5, size=1, prob=predict_no_draw(m5))
  
  m6 = c('France', 'Switzerland')
  m6_winner = sample(m6, size=1, prob=predict_no_draw(m6))
  
  m7 = c('England', 'Germany')
  m7_winner = sample(m7, size=1, prob=predict_no_draw(m7))
  
  m8 = c('Sweden', 'Ukraine')
  m8_winner = sample(m8, size=1, prob=predict_no_draw(m8))
  
  #Quarter finals
  qf1 = c(m6_winner, m5_winner)
  qf1_winner = sample(qf1, size=1, prob=predict_no_draw(qf1))
  
  qf2 = c(m4_winner, m2_winner)
  qf2_winner = sample(qf2, size=1, prob=predict_no_draw(qf2))
  
  qf3 = c(m3_winner, m1_winner)
  qf3_winner = sample(qf3, size=1, prob=predict_no_draw(qf3))
  
  qf4 = c(m8_winner, m7_winner)
  qf4_winner = sample(qf4, size=1, prob=predict_no_draw(qf4))
  
  #Semi final
  
  sf1 = c(qf2_winner, qf1_winner)
  sf1_winner = sample(sf1, size=1, prob=predict_no_draw(sf1))
  
  sf2 = c(qf4_winner, qf3_winner)
  sf2_winner = sample(sf2, size=1, prob=predict_no_draw(sf2))
  
  winner = sample(c(sf1_winner, sf2_winner), size=1, prob = predict_no_draw(c(sf1_winner, sf2_winner)))
  
  tibble(
    m1_winner,
    m2_winner,
    m3_winner,
    m4_winner,
    m5_winner,
    m6_winner,
    m7_winner,
    m8_winner,
    qf1_winner,
    qf2_winner,
    qf3_winner,
    qf4_winner,
    sf1_winner,
    sf2_winner,
    winner
  )
  
}




results = readRDS('predictions/ro16.RDS')

remaining_teams = tibble(team = c('Wales','Denmark', 'Italy','Austria', 'Netherlands','Czech', 'Belgium','Portugal', 'Croatia','Spain', 'France', 'Switzerland', 'England', 'Germany', 'Sweden', 'Ukraine'))


results %>% 
  filter(m4_winner != 'Belgium') %>% 
  select(winner) %>% 
  count(winner) %>% 
  right_join(remaining_teams, by = c('winner'= 'team')) %>% 
  mutate(p = n/sum(n)) %>% 
  ggplot(aes(p, forcats::fct_reorder(winner, p)))+
  geom_col()+
  scale_x_continuous(labels = scales::percent)+
  theme(panel.grid.major = element_line())+
  labs(x = 'Probability of Winning Touranment', y = '')




