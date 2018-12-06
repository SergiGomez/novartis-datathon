

library(ggplot2)
library(data.table)
library(prophet)






get_ts <- function(dt, ts_mode = "multiplicative", n_ahead = 12, ts_freq = "month"){
  
  dt <- data.table(dt)
  
  m <- prophet(history, seasonality.mode = ts_mode)
  future = make_future_dataframe(m, periods = n_ahead, freq = ts_freq)
  forecast = data.table(predict(m, future))
  
  # plot(m, forecast)
  # prophet_plot_components(m, forecast)
  
  forecast[, ds := as.character(ds)]
  dt[, ds := as.character(ds)]
  
  dt_res <- merge(forecast, dt, by = "ds", all.x = T)
  
  
  if(ts_mode == "multiplicative"){
    dt_res[, remainder := y/yhat]
  } else {
    dt_res[, remainder := y - yhat]
  }
  
  
  return(dt_res)
  
}


# Example
dt <- data.frame(ds = seq(as.Date('2012-01-01'), as.Date('2017-12-01'), by = 'm'))
n <- nrow(history)
dt$y <- 100 + sin(1:n/200) + rnorm(n)/10

plot(history, t = "l")


res <- get_ts(dt)
# La columna "remainder" es la serie temporal quitando trend & seasonality

