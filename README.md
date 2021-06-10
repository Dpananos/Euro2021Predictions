# Euro 2021 Predictions

## Methods

The model is taken almost directly from [Andrew Gelman's blog](https://statmodeling.stat.columbia.edu/2014/07/13/stan-analyzes-world-cup-data/).  Andrew does a better job of explaining it there, but I will do my best.

The model starts with assuming that team $j$ has some latent ability $a_j$.  This latent ability is heirarchically modelled so that

$$ a_j \sim \mathcal{N}(\beta \mbox{pts}, \sigma^2_a) $$

Here, $\mbox{pts}$ is   prior information about each team's ranking prior to the tournament starting (more on that in the data section). In essence, ability is directly related to team ranking, and the strength of that relationship is $\beta$.

The outcome is not goals per team but rather the difference in goals.  If team $i$ scores $y_i$ goals and team $j$ scores $y_j$ goals, then we assume

$$ \operatorname{sgn}(y_i - y_j) \sqrt{y_i - y_j} \sim \operatorname{Student-t}(\nu, a_i - a_j, \sigma^2_y)  $$

Here, I am following Stan convention and parameterizing the Student t by its degrees of freedom ($\nu$), non-centrality parameter ($a_i - a_j$), and something that looks like the variance ($\sigma^2_y$).

What this means is that if team $i$ has more ability that team $j$, then the difference in number of goals will be in their favor.  The square root is included to discount complete blowouts for giving too much ability to any one team (remember Brazil's 8-0 defeat at the hands of Germany).  The model code and model priors can be found in `models/`.

## Data

I recorded all qualifying match data from [here](https://www.uefa.com/uefaeuro-2020/).  Prior ranking information is found [here](https://www.fifa.com/fifa-world-ranking/ranking-table/men/#UEFA).  I use the `Total Points` column as the `pts` variable in the model rather than using ranking as Gelman does.  I do standardize this column however (dividing by 2 standard deviations as Gelman does).

I will periodically update the data as games are played.