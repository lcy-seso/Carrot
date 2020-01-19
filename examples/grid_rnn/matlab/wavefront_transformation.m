clc;
clear all;
close all;
%%

hold on
grid on

depth = 3;

plot_a_sample(depth,4,7,'b', 'b')

% plot_a_sample(depth,3,5,'g', 'g')
% plot_a_sample(depth, 6, 9, 'b', 'b')
%%

% z = ones(P, N) * (M - 1);
% mesh(x,y,z)

%% perpendicular to x axis
% [y, z] = meshgrid(1:P, 0:M);
% x = zeros(M, P);
% mesh(x,y,z)
% 
% x = N * ones(M, P);
% mesh(x,y,z)
% 
% %% perpendicular to y axis
% [x, z] = meshgrid(1:N, 0:M-1);
% y = zeros(M, N);
% mesh(x,y,z)
% 
% [x, z] = meshgrid(1:N, 0:M-1);
% y = P * ones(M, N);
% mesh(x,y,z)

% scatter3(xs,ys,zs);
% mesh(xs, ys, zs);
% plot3(xs,ys,zs);
