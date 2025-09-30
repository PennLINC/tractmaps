library(tidyr)
library(mgcv)
library(gratia)
library(tidyverse)
library(dplyr)

#### FIT GAM ####
##Function to fit a GAM (measure ~ s(smooth_var, k = knots, fx = set_fx) + covariates)) 
## per each tract and save out statistics and derivative-based characteristics
gam.fit.smooth <- function(measure, dataset, region, smooth_var, 
                           covariates, knots, set_fx = FALSE, 
                           stats_only = FALSE)
  {
  #Fit the gam
  dataname <- sprintf("%s.%s", measure, dataset) 
  gam.data <- get(dataname)
  parcel <- region
  region <- str_replace(region, "-", ".")
  modelformula <- as.formula(sprintf("%s ~ s(%s, k = %s, fx = %s) + %s", 
                                     region, smooth_var, knots, set_fx, 
                                     covariates))
  gam.model <- gam(modelformula, method = "REML", data = gam.data)
  gam.results <- summary(gam.model)
  
  #GAM derivatives
  #Get derivatives of the smooth function using finite differences
  #derivative at 200 indices of smooth_var with a simultaneous CI
  derv <- derivatives(gam.model, select = sprintf('s(%s)',smooth_var), 
                      interval = "simultaneous", unconditional = F)
  
  #Identify derivative significance window(s)
  #add "sig" column (TRUE/FALSE) to derv
  derv <- derv %>%
     # remove the . from the start of the column names
    rename_with(~ gsub("^\\.", "", .x), starts_with(".")) %>%
    # rename the smooth variable column to "data" for consistency
    rename(data = !!smooth_var) %>%
    # rename lower_ci to lower and upper_ci to upper
    rename(lower = lower_ci, upper = upper_ci) %>%
    #derivative is sig if the lower CI is not < 0 while the upper CI is > 0
    # (i.e., when the CI does not include 0)
    mutate(sig = !(0 > lower & 0 < upper))
  #add "sig_deriv derivatives column where non-significant derivatives are set to 0
  derv$sig_deriv = derv$derivative*derv$sig
  
  #GAM statistics
  #F value for the smooth term and GAM-based significance of the smooth term
  gam.smooth.F <- gam.results$s.table[3]
  gam.smooth.pvalue <- gam.results$s.table[4]
  
  #Calculate the magnitude and significance of the smooth term effect 
  #by comparing full and reduced models
  ##Compare a full model GAM (with the smooth term) to a nested, reduced model 
  ##(with covariates only)
  nullmodel <- as.formula(sprintf("%s ~ %s", region, covariates)) #no smooth term
  gam.nullmodel <- gam(nullmodel, method = "REML", data = gam.data)
  gam.nullmodel.results <- summary(gam.nullmodel)
  
  ##Full versus reduced model anova p-value
  anova.smooth.pvalue <- anova.gam(gam.nullmodel,
                                   gam.model,
                                   test='Chisq')$`Pr(>Chi)`[2]
  
  ##Full versus reduced model direction-dependent partial R squared
  ### effect size
  sse.model <- sum((gam.model$y - gam.model$fitted.values)^2)
  sse.nullmodel <- sum((gam.nullmodel$y - gam.nullmodel$fitted.values)^2)
  partialRsq <- (sse.nullmodel - sse.model)/sse.nullmodel
  ### effect direction
  mean.derivative <- mean(derv$derivative)
  #if the average derivative is less than 0, make the effect size estimate negative
  if(mean.derivative < 0){
    partialRsq <- partialRsq*-1}
  
  #Derivative-based temporal characteristics
  #Age of developmental change onset
  #if derivative is significant at at least 1 age, find first age in the smooth 
  #where derivative is significant
  if(sum(derv$sig) > 0){
    change.onset <- min(derv$data[derv$sig==T])}
  #if gam derivative is never significant, assign NA
  if(sum(derv$sig) == 0){
    change.onset <- NA}
  
  #Age of maximal developmental change
  if(sum(derv$sig) > 0){ 
    #absolute value significant derivatives
    derv$abs_sig_deriv = round(abs(derv$sig_deriv),5)
    #find the largest derivative
    maxval <- max(derv$abs_sig_deriv)
    #identify the age(s) at which the derivative is greatest in absolute magnitude
    window.peak.change <- derv$data[derv$abs_sig_deriv == maxval]
    #identify the age of peak developmental change
    peak.change <- mean(window.peak.change)}
  if(sum(derv$sig) == 0){ 
    peak.change <- NA}  
  
  #Age of decrease onset
  #identify all ages with a significant negative derivative 
  #(i.e., smooth_var indices where y is decreasing)
  if(sum(derv$sig) > 0){ 
    decreasing.range <- derv$data[derv$sig_deriv < 0]
    if(length(decreasing.range) > 0)
      #find youngest age with a significant negative derivative
      decrease.onset <- min(decreasing.range)
    if(length(decreasing.range) == 0)
      decrease.onset <- NA}
  if(sum(derv$sig) == 0){
    decrease.onset <- NA}  
  
  #Age of increase offset
  ##identify all ages with a significant positive derivative 
  #(i.e., smooth_var indices where y is increasing)
  if(sum(derv$sig) > 0){ 
    increasing.range <- derv$data[derv$sig_deriv > 0]
    if(length(increasing.range) > 0)
      #find oldest age with a significant positive derivative
      increase.offset <- max(increasing.range)
    if(length(increasing.range) == 0)
      increase.offset <- NA}
  if(sum(derv$sig) == 0){ 
    increase.offset <- NA}  
  
  #Age of maturation
  #find last age in the smooth where derivative is significant
  if(sum(derv$sig) > 0){ 
    change.offset <- max(derv$data[derv$sig==T])}
  if(sum(derv$sig) == 0){ 
    change.offset <- NA}  
  
  full.results <- cbind(parcel, gam.smooth.F, gam.smooth.pvalue, partialRsq, 
                        anova.smooth.pvalue, change.onset, peak.change, 
                        decrease.onset, increase.offset, change.offset)
  stats.results <- cbind(parcel, gam.smooth.F, gam.smooth.pvalue, partialRsq, 
                         anova.smooth.pvalue)
  if(stats_only == TRUE)
    return(stats.results)
  if(stats_only == FALSE)
    return(full.results)
}

##Function to fit a GAM (measure ~ s(smooth_var, k = knots, fx = set_fx) + linear term + covariates)) 
gam.fit.linear <- function(measure, dataset, region, smooth_var, 
                           linear_var, covariates, knots, set_fx = FALSE)
  {
  #Fit the gam
  dataname <- sprintf("%s.%s", measure, dataset)
  gam.data <- get(dataname)
  parcel <- region
  region <- str_replace(region, "-", ".")
  modelformula <- as.formula(sprintf("%s ~ %s + s(%s, k = %s, fx = %s) + %s", 
                                     region, linear_var, smooth_var, knots, 
                                     set_fx, covariates))
  # modelformula <- as.formula(sprintf("%s ~ s(%s, k = %s, fx = %s) + 
  #                                    s(%s, k = %s, fx = %s) + ti(%s, %s) + %s", 
  #                                    region, linear_var, knots, set_fx, 
  #                                    smooth_var, knots, set_fx, linear_var, 
  #                                    smooth_var, covariates))
  gam.model <- gam(modelformula, method = "REML", data = gam.data)
  gam.results <- summary(gam.model)

  #GAM statistics
  #F value for the linear term and GAM-based significance of the linear term
  gam.linear.F <- gam.results$pTerms.table[1,2]
  gam.linear.pvalue <- gam.results$pTerms.table[1,3]

  #Calculate the magnitude and significance of the linear term effects 
  #by comparing full and reduced models
  ##Compare a full model GAM (with the linear term) to a nested, 
  #reduced model (with covariates only + smooth var)
  nullmodel <- as.formula(sprintf("%s ~ s(%s, k = %s, fx = %s) + %s", region, 
                                  smooth_var, knots, set_fx, covariates))
  gam.nullmodel <- gam(nullmodel, method = "REML", data = gam.data)
  gam.nullmodel.results <- summary(gam.nullmodel)

  ##Full versus reduced model anova p-value
  anova.linear.pvalue <- anova.gam(gam.nullmodel,
                                   gam.model, 
                                   test='Chisq')$`Pr(>Chi)`[2]
  #if residual deviance is exactly equal between full and reduced models and 
  # p=value = NA, set p = 1
  if(is.na(anova.linear.pvalue)){
    anova.linear.pvalue <- 1}

  ##Full versus reduced model direction-dependent partial R squared
  ### effect size
  sse.model <- sum((gam.model$y - gam.model$fitted.values)^2)
  sse.nullmodel <- sum((gam.nullmodel$y - gam.nullmodel$fitted.values)^2)
  partialRsq <- (sse.nullmodel - sse.model)/sse.nullmodel
  ### effect direction
  #if the coefficient (or tvalue) for linear term is less than 0, 
  #make the effect size estimate negative
  if(gam.results$p.table[2,3] < 0){
    partialRsq <- partialRsq*-1}

  full.results <- cbind(parcel, gam.linear.F, gam.linear.pvalue, 
                        partialRsq, anova.linear.pvalue)

  return(full.results)
}

#### PREDICT GAM FITTED VALUES ####
##Function to predict fitted values of a measure based on a fitted GAM smooth 
##(measure ~ s(smooth_var, k = knots, fx = set_fx) + covariates)) and a prediction df
gam.smooth.predict <- function(measure, atlas, dataset, region, smooth_var, 
                               covariates, knots, set_fx = FALSE, increments)
  {
  #Fit the gam
  dataname <- sprintf("%s.%s.%s", measure, atlas, dataset) 
  gam.data <- get(dataname)
  parcel <- region
  region <- str_replace(region, "-", ".")
  modelformula <- as.formula(sprintf("%s ~ s(%s, k = %s, fx = %s) + %s", 
                                     region, smooth_var, knots, set_fx, 
                                     covariates))
  gam.model <- gam(modelformula, method = "REML", data = gam.data)
  gam.results <- summary(gam.model)
  
  #Extract gam input data
  #extract the data used to build the gam, i.e., a df of y + predictor values 
  df <- gam.model$model
  
  #Create a prediction data frame
  #number of predictions to make; predict at np increments of smooth_var
  np <- increments
  #initiate a prediction df 
  thisPred <- data.frame(init = rep(0,np))
  
  #gam model predictors (smooth_var + covariates)
  theseVars <- attr(gam.model$terms,"term.labels")
  #classes of the model predictors and y measure
  varClasses <- attr(gam.model$terms,"dataClasses")
  #the measure to predict
  thisResp <- as.character(gam.model$terms[[2]])
  #fill the prediction df with data for predictions. These data will be used 
  #to predict the output measure (y) at np increments of the smooth_var, 
  #holding other model terms constant
  for (v in c(1:length(theseVars))) {
    thisVar <- theseVars[[v]]
    thisClass <- varClasses[thisVar]
    #generate a range of np data points, from min of smooth term to max of smooth term
    if (thisVar == smooth_var) { 
      thisPred[,smooth_var] = seq(min(df[,smooth_var],na.rm = T),
                                  max(df[,smooth_var],na.rm = T), 
                                  length.out = np)
    } 
    else {
      switch (thisClass,
              #make predictions based on median value
              "numeric" = {thisPred[,thisVar] = median(df[,thisVar])},
              #make predictions based on first level of factor 
              "factor" = {thisPred[,thisVar] = levels(df[,thisVar])[[1]]},
              #make predictions based on first level of ordinal variable
              "ordered" = {thisPred[,thisVar] = levels(df[,thisVar])[[1]]}
      )
    }
  }
  pred <- thisPred %>% select(-init)
  
  #Generate predictions based on the gam model and predication data frame
  predicted.smooth <- fitted_values(object = gam.model, data = pred)
  predicted.smooth <- predicted.smooth %>% select(all_of(smooth_var), .fitted, 
                                                  .se, .lower_ci, .upper_ci)
  
  #Determine the smooth_var index at which y is maximal, based on the predicted smooth
  maxidx <- which.max(predicted.smooth$.fitted)
  peak <- predicted.smooth[maxidx,smooth_var]

  smooth.fit <- list(parcel, peak[[1]], predicted.smooth)
  return(smooth.fit)
}

##Function to predict fitted values of a measure based on a fitted GAM with linear var
##(measure ~ s(smooth_var, k = knots, fx = set_fx) + linearvar + covariates)) and a prediction df
gam.linear.predict <- function(measure, atlas, dataset, region, smooth_var, 
                               linear_var, covariates, knots, 
                               set_fx = FALSE, increments)
{
  #Fit the gam
  dataname <- sprintf("%s.%s.%s", measure, atlas, dataset) 
  gam.data <- get(dataname)
  parcel <- region
  region <- str_replace(region, "-", ".")
  modelformula <- as.formula(sprintf("%s ~ %s + s(%s, k = %s, fx = %s) + %s", 
                                     region, linear_var, smooth_var, knots, 
                                     set_fx, covariates))
  gam.model <- gam(modelformula, method = "REML", data = gam.data)
  gam.results <- summary(gam.model)
  
  #Extract gam input data
  #extract the data used to build the gam, i.e., a df of y + predictor values 
  df <- gam.model$model
  
  #Create a prediction data frame
  #number of predictions to make; predict at np increments of linear_var
  np <- increments
  #initiate a prediction df 
  thisPred <- data.frame(init = rep(0,np))
  
  #gam model predictors (smooth_var + linear_var + covariates)
  theseVars <- attr(gam.model$terms,"term.labels")
  #classes of the model predictors and y measure
  varClasses <- attr(gam.model$terms,"dataClasses")
  #the measure to predict
  thisResp <- as.character(gam.model$terms[[2]])
  #fill the prediction df with data for predictions. These data will be used 
  #to predict the output measure (y) at np increments of the linear_var, 
  #holding other model terms constant
  for (v in c(1:length(theseVars))) {
    thisVar <- theseVars[[v]]
    thisClass <- varClasses[thisVar]
    #generate a range of np data points, from min of linear term to max of linear term
    if (thisVar == linear_var) { 
      thisPred[,linear_var] = seq(min(df[,linear_var],na.rm = T),
                                  max(df[,linear_var],na.rm = T), 
                                  length.out = np)
    } 
    else {
      switch (thisClass,
              #make predictions based on median value
              "numeric" = {thisPred[,thisVar] = median(df[,thisVar])},
              #make predictions based on first level of factor 
              "factor" = {thisPred[,thisVar] = levels(df[,thisVar])[[1]]},
              #make predictions based on first level of ordinal variable
              "ordered" = {thisPred[,thisVar] = levels(df[,thisVar])[[1]]}
      )
    }
  }
  pred <- thisPred %>% select(-init)
  
  #Generate predictions based on the gam model and predication data frame
  predicted.linear <- fitted_values(object = gam.model, data = pred)
  predicted.linear <- predicted.linear %>% select(all_of(linear_var), .fitted, 
                                                  .se, .lower_ci, .upper_ci)
  
  #Determine the linear_var index at which y is maximal, based on the predicted linear
  maxidx <- which.max(predicted.linear$.fitted)
  peak <- predicted.linear[maxidx,linear_var]

  linear.fit <- list(parcel, peak[[1]], predicted.linear)
  return(linear.fit)
}

