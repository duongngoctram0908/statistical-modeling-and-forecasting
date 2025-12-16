#### Bai thuc hanh 1 ----
#install.packages("car")
library(car)
setwd("C:/Users/Admin/Desktop/Dự báo")
data <- read.csv("final-project-30firms.csv")
data_sub <- subset(data, select = c("Y","X1","X2","X3","X4","X5","X6"))
## Scatter matrix plot
plot(data_sub, col = "dodgerblue", lwd = 1.2, cex = 1.5)
## Model with all independent variables
mylm1 <- lm(Y ~ X1+X2+X3+X4+X5+X6, data = data_sub)
summary(mylm1)
## Best model
mylm2 <- lm(Y ~ X5+X6, data = data_sub)
summary(mylm2)
## Multicollinearity
vif_values <- vif(mylm1)
## Residuals analyst
mylm_resid_std <- rstandard(mylm2)
y_hat <- mylm2$fit
# QQ-plot
qqnorm(mylm_resid_std, col = "green3", cex = 1.2, lwd = 1.5)
qqline(mylm_resid_std, col = "orange3", cex = 1.2, lwd = 1.5)
# Residuals plot
plot(y_hat, mylm_resid_std, col = "green3", cex = 1.2, lwd = 1.5,
     xlab = "Predicted Reponse", ylab = "Plot residuals")
abline(h = 0, col = "red", lty = 2, lwd = 1.5) # y = 0
## Cook's distance
plot(cooks.distance(mylm2), type = "h", col = "purple", lwd = 1.5)




#### Bai thuc hanh 3 ----
library(astsa)
setwd("C:/Users/Admin/Desktop/[DB]_N2_FinalProject/Data")
data <- read.csv("Motor-vehicle_deaths_2018-2023.csv")
myts <- ts(data$Deaths, start = c(2018,1), end = c(2023,12), frequency = 12)
## Data visualization
ts.plot(myts, col = "dodgerblue", lwd = 2.5, 
        main = "Motor Vehicle Monthly Deaths 2018 - 2023", 
        xlab = "Year", ylab = "Deaths")
grid(15, 15, col = "gray")
## Monthplot
monthplot(myts, col = "dodgerblue", lwd = 2.5, col.base = "indianred", xlab = "Month", ylab = "Deaths")
grid(15, 15, col = "gray")
## Difference (eliminating trend)
dx <- diff(myts)
plot(dx, xlab = "Year", col = "dodgerblue", lwd = 2.5)
grid(15, 15, col = "gray")
# Difference D = 12 (eliminating seasonality)
ddx <- diff(diff(myts), 12)
plot(ddx, col = "dodgerblue", lwd = 2.5, xlab = "Year", ylab = "ddx")
grid(15, 15, col = "gray")
## Choose the SARIMA model
acf2(ddx, 48)
sarima(myts, 2, 1, 0, 1, 1, 1, 12)
sarima(myts, 0, 1, 2, 1, 1, 1, 12)
sarima(myts, 2, 1, 2, 1, 1, 1, 12)
## Best model
model <- sarima(myts, 0, 1, 2, 1, 1, 1, 12)
## Residual plots
residual <- sarima(myts, 0, 1, 2, 1, 1, 1, 12)$fit$resid
plot(residual, col = "dodgerblue", lwd = 2.5,
     main = "Residuals", xlab = "Time", ylab = "")
grid(10, 10, "gray")
acf2(residual)
## Forecasting for the next 12 months
sarima.for(myts, 0, 1, 2, 1, 1, 1, 12, n.ahead = 12)
# Explaining why are the confidence intervals so large? (SARIMA)
fitted_values <- data$Deaths - model$fit$residuals
data_ex <- ts(data.frame(data$Deaths, fitted_values), start = 2018, end = 2023, frequency = 12)
plot(data_ex, plot.type = "single", col = c("dodgerblue", "red"), lwd = 2.5,
     main = "Observed and Predicted values",
     ylab = "Deaths",
     xlab = "Time")
grid(10, 10, "gray")
legend(x = 2018, y = 4250, legend = c("Observed values", "Predicted values"), 
       col = c("dodgerblue", "red"), lty = 1, lwd = 2.5, 
       border = NULL, bty = "n", cex = 0.8,
       x.intersp = 0.5, y.intersp = 1, text.width = 1.5)