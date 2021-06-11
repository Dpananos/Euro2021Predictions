data {
  int nteams;
  int ngames;
  int team1[ngames];
  int team2[ngames];
  vector[ngames] score1;
  vector[ngames] score2;
  vector[nteams] prior_score;
  real b_mean;
  real b_sd;
}
transformed data {
  vector[ngames] dif;
  vector[ngames] sqrt_dif;
  real score_sgn;
  dif = score1 - score2;
}
parameters {
  real<lower=0> b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
  vector[nteams] eta_a;
  real<lower=0> df;
}
transformed parameters {
  vector[nteams] a;
  a = b*prior_score + sigma_a*eta_a;
}  
model {
  b ~ normal(b_mean, b_sd);
  // df ~ gamma(2,0.1);
  df ~ gamma(5,0.5);
  eta_a ~ normal(0,1);
  for (i in 1:ngames)
    dif[i] ~ student_t(df, a[team1[i]]-a[team2[i]], sigma_y);
}
generated quantities{
  real yppc[ngames];
  for (i in 1:ngames){
    
    yppc[i] = round(student_t_rng(df, a[team1[i]]-a[team2[i]], sigma_y ));
  }
  
}
