# FT_LINEAR_REGRESSION
Project aiming to code Batch Gradient Descant algorithm


### What is it ?
Gradient descent algorithms aim to find the best parameters (wight/theta)
that minimize the loss function (error).
Do to that we'll use the batch gradient descent algorithm that will
go throught all data at each iteration of the algorithm.
This method is more stable and accurate but also slower.
We can use it in this case due to the small size of out dataset (24 inputs).

### How does it work ?
In our case we want to predict the price (output) based on the km (feature).
So the prediction h(X) is **price = θ0 + (θ1 * km)**

In order to make the algorithm faster (higher learning rate),
we'll normalize the 'km' (feature) using Z-score scaling (standardization).
**normalized_x = (x - mean) / std**

Here the gradient formulas are given so we can skip (even if it matters)
the loss function.
**θ0 = θ0 - α * (1/m) * ∑(h(X) - price(X))**
**θ1 = θ1 - α * (1/m) * ∑((h(X) - price(X)) * km(X))**
With α the learning rate / m the dataset size / θ the feature.

After this we 'unormalize' theta again and voila, we updated our weights!
