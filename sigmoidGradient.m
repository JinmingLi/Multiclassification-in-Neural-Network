function g = sigmoidGradient(z)

h = sigmoid(z);
g = h .* (1-h);

end
