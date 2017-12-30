function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X = [ones(m, 1), X]; % add ones m*(n+1)
a1 = X;
z2 = Theta1 * X' ; 
a2 = sigmoid(z2);

a2 = [ones(m, 1), a2']; 
z3 = Theta2 * a2'; 
a3 = sigmoid(z3);

h = a3; % m*k
y_temp = zeros(num_labels, m); 

for i = 1:m
y_temp(y(i), i) = 1;
end

part1 = y_temp .* log(h);
part2 = (1-y_temp) .* log((1-h));
sum1 = sum(sum(-part1 - part2));
J_ori = sum1 / m;

% regularized cost function
punish_Theta1 = sum(sum(Theta1(:, 2:end).^2));
punish_Theta2 = sum(sum(Theta2(:, 2:end).^2));

J = J_ori + lambda/2/m*(punish_Theta1 + punish_Theta2);

for t = 1:m
a1 = X(t, :);
z2 = Theta1 * a1';
a2 = sigmoid(z2);
a2 = [1; a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

z2 = [1;  z2];
 
delta3 = a3 - y_temp(:, t);
delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);
delta2 = delta2(2:end);

Theta2_grad = Theta2_grad + delta3 * a2';
Theta1_grad = Theta1_grad + delta2 * a1;
end
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

reg_theta1 = Theta1(:, 2:end) * lambda/m;
reg_theta2 = Theta2(:, 2:end) * lambda/m;


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + reg_theta1;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + reg_theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
