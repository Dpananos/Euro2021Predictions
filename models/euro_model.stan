data {
  int nteams;
  int ngames;
  int team1[ngames];
  int team2[ngames];
  vector[ngames] score1;
  vector[ngames] score2;
  real df;
}
transformed data {
  vector[ngames] dif;
  vector[ngames] sqrt_dif;
  real score_sgn;
  dif = score1 - score2;
  for (i in 1:ngames){
    score_sgn = (dif[i]<0) ? -1 : 1;
    sqrt_dif[i] = score_sgn*sqrt(fabs(dif[i]));
  }
    
}
parameters {
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
  vector[nteams] eta_a;
}
transformed parameters {
  vector[nteams] a;
  a = sigma_a*eta_a;
}  
model {
  eta_a ~ normal(0,1);
  for (i in 1:ngames)
    sqrt_dif[i] ~ student_t(df, a[team1[i]]-a[team2[i]], sigma_y);
}
generated quantities{
  real yppc[ngames];
  real sgn;
  for (i in 1:ngames){
    sgn = (a[team1[i]] < a[team2[i]]) ? -1 : 1;
    yppc[i] = sgn*round(pow(student_t_rng(df, a[team1[i]]-a[team2[i]], sigma_y ),2));
  }
    
}