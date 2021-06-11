library(tidyverse)

file = read_delim('data/euro_results_2016.txt', delim = '\n', col_names = c('data'))

file %>% 
  mutate(
    results = {
      data %>% 
        str_squish() %>% 
        str_replace(pattern = 'Czech Republic', replacement = 'Czech') %>% 
        str_replace(pattern = 'Republic of Ireland', replacement = 'Ireland') %>% 
        str_replace(pattern = 'Northern Ireland', replacement = 'NorthernIreland') %>% 
        str_replace(pattern = 'Faroe Islands', replacement = 'FaroeIslands') %>% 
        str_replace(pattern = 'San Marino', replacement = 'SanMarino') %>% 
        str_replace(pattern = 'Bosnia-Herzegovina', replacement = 'Bosnia') %>% 
        str_replace(pattern = 'Russia, awarded due to crowd trouble', replacement = 'Russia') %>% 
        str_replace(pattern = 'Albania forfeits, but Serbia deducted 3 points for crowd trouble', replacement = 'Albania') %>% 
        str_replace(pattern = 'Kazakhtan', replacement = 'Kazakhstan') %>% 
        str_replace(pattern = 'FYR Macedonia', replacement = 'Macedonia')
    }
    
  )  %>% 
  separate(results, c('team1', 'score1', 'score2', 'team2'), ) %>% 
  select(-data) %>% 
  write_csv('data/2016_cleaned.csv')