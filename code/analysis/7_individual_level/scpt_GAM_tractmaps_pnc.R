# ------------------------------------------------------------
# Script to run GAMs on PNC data
# ------------------------------------------------------------

# This script runs GAMs on PNC data to assess the relationship between tract scalar measures, age, and cognition.
# It runs the func_GAM_tractmaps.R functions to fit the GAMs.
# Inputs: PNC FA, age and cognition csvs. 
# Outputs: Partial R2 statistics for each tract and each variable.

# ------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------

rm(list = ls())
library(stringr)
library(tidyr)
library(mgcv)
library(gratia)
library(tidyverse)
library(dplyr)
library(ggseg)
library(paletteer)
library(pals)
library(ggseg3d)
library(ggplot2)
library(scales)

# ------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------

# set root directory
root <- "/Users/joelleba/PennLINC/tractmaps"

# source GAM functions
source(file.path(root, "code/analysis/7_individual_level/func_GAM_tractmaps.R"))

# set data path
data_path <- file.path(root, "data/derivatives/individual_level_pnc/final_sample/")

# set output path
outpath <- file.path(root, "results/individual_level/")

dataset <- 'pnc'
gamtype <- 'age' # 'age' or 'cognition'

metric <- 'fa'

if (gamtype == 'age'){
  dtype <- sprintf('%s_final_sample_%s', dataset, metric)
}
if (gamtype == 'cognition'){
  dtype <- sprintf('%s_final_cognition_sample_%s', dataset, metric)
}

if (gamtype == 'age'){smooth_var <- 'age'}
if (gamtype == 'age'){covars <- 'sex + mean_fd'}

if (gamtype == 'cognition'){smooth_var <- 'age'}
if (gamtype == 'cognition'){covars <- 'sex + mean_fd'}
if (gamtype == 'cognition'){linear_var <- 'F3_Executive_Efficiency'} 


# prepare data for GAMs
metric.all <- read.csv(paste(data_path, sprintf('%s.csv', dtype), sep = ""), sep = ',')

# will use dataframe as a covariate
metric.all$dataset <- as.factor(dataset)
# will use sex as a covariate
metric.all$sex <- as.factor(metric.all$sex)

# ------------------------------------------------------------
# --- Fit GAMs ---
# ------------------------------------------------------------

if(gamtype=='age'){
  # list of tracts to run gam.fit.smooth function on below
  tract_labels <- names(metric.all)[sapply(names(metric.all), function(x) str_detect(x, "Association|Projection"))] %>%
    data.frame(tract = .) # get column names containing 'Association' or 'Projection'
  # count number of tracts
  n_tracts <- nrow(tract_labels)
  gam.variable.tract <- matrix(data=NA, nrow=n_tracts, ncol=10)
  # for each tract
  for(row in c(1:n_tracts)){
    tract <- tract_labels$tract[row]
    GAM.RESULTS <- gam.fit.smooth(measure = "metric", 
                                  dataset = "all", 
                                  region = tract, smooth_var = smooth_var, 
                                  covariates = covars, 
                                  knots = 3, set_fx = FALSE, stats_only = FALSE)
    # and append results to output df 
    gam.variable.tract[row,] <- GAM.RESULTS}
  
  gam.variable.tract <- as.data.frame(gam.variable.tract)
  colnames(gam.variable.tract) <- c("tract","GAM.variable.Fvalue","GAM.variable.pvalue",
                                       "GAM.variable.partialR2","Anova.variable.pvalue",
                                       "age.onsetchange", "age.peakchange",
                                       "minage.decrease","maxage.increase","age.maturation")
  cols = c(2:10)    
  gam.variable.tract[,cols] = apply(gam.variable.tract[,cols], 2, 
                                       function(x) as.numeric(as.character(x)))
}

if(gamtype=='cognition'){
  # list of tracts to run gam.fit.smooth function on below
  tract_labels <- names(metric.all)[sapply(names(metric.all), function(x) str_detect(x, "Association|Projection"))] %>%
    data.frame(tract = .) # get column names containing 'Association' or 'Projection'
  # count number of tracts
  n_tracts <- nrow(tract_labels)
  gam.variable.tract <- matrix(data=NA, nrow=n_tracts, ncol=5)
  # for each tract
  for(row in c(1:n_tracts)){
    tract <- tract_labels$tract[row]
    GAM.RESULTS <- gam.fit.linear(measure = "metric",
                                  dataset = "all",
                                  region = tract,
                                  smooth_var = smooth_var,
                                  linear_var = linear_var,
                                  covariates = covars,
                                  knots = 3, set_fx = FALSE)
    # and append results to output df
    gam.variable.tract[row,] <- GAM.RESULTS}

  gam.variable.tract <- as.data.frame(gam.variable.tract)
  colnames(gam.variable.tract) <- c("tract","GAM.variable.Fvalue","GAM.variable.pvalue",
                                       "GAM.variable.partialR2","Anova.variable.pvalue")
  cols = c(2:5)
  gam.variable.tract[,cols] = apply(gam.variable.tract[,cols], 2,
                                    function(x) as.numeric(as.character(x)))
  gamtype <- linear_var
}

# ------------------------------------------------------------
# --- FDR correction ---
# ------------------------------------------------------------

csvR2 <- data.frame(gam.variable.tract$tract)
csvR2$partialR2 <- gam.variable.tract$GAM.variable.partialR2

## pvalues
# GAMs
pvalues = gam.variable.tract$GAM.variable.pvalue
GAMpvaluesfdrs<-p.adjust(pvalues, method="BH") # Benjamini-Hochberg FDR correction

# Anova
pvalues = gam.variable.tract$Anova.variable.pvalue
Anovapvaluesfdrs<-p.adjust(pvalues, method="BH") # Benjamini-Hochberg FDR correction

csvR2$anovaPvaluefdr <- Anovapvaluesfdrs
csvR2$gamPvaluefdr <- GAMpvaluesfdrs

# ------------------------------------------------------------
# --- Save results ---
# ------------------------------------------------------------

outputPath <- paste(outpath, sprintf('%s_%s_partialR2_stats.csv', dtype, gamtype), sep="")
write.csv(csvR2, outputPath, row.names=FALSE)

print(paste("Partial R2 stats saved to:", outputPath))
