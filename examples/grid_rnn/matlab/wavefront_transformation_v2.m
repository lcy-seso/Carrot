clc;
clear all;
close all;

%% plot the original cube whose all faces are perpendicular to the axis.

figure()
view(3)
box on
hold on
grid on

title('Iteration Space (Each cube is a sample)')
xlabel('Source Sequence')
ylabel('Target Sequence')
zlabel('Depth')

%% constants
depth = 3;    % z axis
src_len = 5;  % x axis
trg_len = 7;  % y axis

%% plot original space

zmin = 0; zmax = depth-1;
xmin = 1; xmax = src_len;
ymin = 1; ymax = trg_len;

plot_one_cube(xmin:1:xmax,...
    ymin:1:ymax,...
    zmin:1:zmax, 'blue');

%% Plot original points
N = depth*src_len*trg_len;
points_x = zeros(N,1);
points_y = zeros(N,1);
points_z = zeros(N,1);
c = 1;
for d=0:depth-1
    for i = 1:src_len
        for j = 1:trg_len
            points_x(c, 1) = i;
            points_y(c, 1) = j;
            points_z(c, 1) = d;
            c = c + 1;
        end
    end
end
scatter3(points_x, points_y, points_z,...
    'Marker','o','LineWidth', 5,...
    'MarkerEdgeColor', 'b',...
    'MarkerFaceColor', 'b')

%% Plot transformed iteration space.
figure()
view(3)

%% Plot the transformed space
title('Transformed Iteration Space.')
zlen = depth + src_len + trg_len;
plot_one_rhombus_cube(xmin, xmax, ymin, ymax, zmin, zmax,'blue')

box on
hold on
grid on

%% Plot hyperplanes.

for zvalue = 2:12
    plot_hyperplane(0,3,1,5, zvalue, 'red')
end

%% Plot transformed points.
scatter3(points_z, points_x, points_x + points_y + points_z,...
    'Marker','+','LineWidth', 4,...
    'MarkerEdgeColor', 'b',...
    'MarkerFaceColor', 'b')
