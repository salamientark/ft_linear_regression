# FT_LINEAR_REGRESSION
Project aiming to code Batch Gradient Descant algorithm


### What is it ?
Gradient descent algorithms aim to find the best parameters (wight/theta)<br>
that minimize the loss function (error).<br>
Do to that we'll use the batch gradient descent algorithm that will<br>
go throught all data at each iteration of the algorithm.<br>
This method is more stable and accurate but also slower.<br>
We can use it in this case due to the small size of out dataset (24 inputs).<br>

### How does it work ?
In our case we want to predict the price (output) based on the km (feature).<br>
So the prediction h(X) is **price = θ0 + (θ1 * km)**<br>

In order to make the algorithm faster (higher learning rate),<br>
we'll normalize the 'km' (feature) using Z-score scaling (standardization).<br>
**normalized_x = (x - mean) / std**<br>

Here the gradient formulas are given so we can skip (even if it matters)<br>
the loss function.<br>
**θ0 = θ0 - α * (1/m) * ∑(h(X) - price(X))**<br>
**θ1 = θ1 - α * (1/m) * ∑((h(X) - price(X)) * km(X))**<br>
With α the learning rate / m the dataset size / θ the feature.<br>

After this we 'unormalize' theta again and voila, we updated our weights!
