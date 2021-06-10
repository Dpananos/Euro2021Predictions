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
  dif = score1 - score2;
  for (i in 1:ngames)
    sqrt_dif[i] = 2*(step(dif[i]) - .5)*sqrt(fabs(dif[i]));
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
    yppc[i] = round(sgn*pow(student_t_rng(df, a[team1[i]]-a[team2[i]], sigma_y ),2));
  }
    
}