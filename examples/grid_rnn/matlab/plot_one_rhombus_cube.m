function [outputArg1,outputArg2] = plot_one_rhombus_cube(x_value,y_value,z_value, color)

x_value_shift = x_value(2,1) - x_value(1,1);
x_value_dim = length(x_value(1,:));
y_value_dim = length(y_value);
z_value_dim = length(z_value);

% 8个顶点分别为：
% 与(0,0,0)相邻的4个顶点
% 与(a,b,c)相邻的4个顶点
a = x_value(1,end);
b = y_value(end);
c = z_value(end);
a_top_1 = x_value(2, 1);
a_top_2 = x_value(2, end);
%V = [0 0 0;a 0 0;0 b 0;0 0 c;
%     a b c;0 b c;a 0 c;a b 0];
V = [0 0 0;a 0 0;0 b 0;a_top_1 0 c;
     a_top_2 b c;a_top_1 b c;a_top_2 0 c;a b 0];
% 6个面
% 以(0,0,0)为顶点的三个面
% 以(a,b,c)为顶点的三个面
F = [1 2 7 4;1 3 6 4;1 2 8 3;
     5 8 3 6;5 7 2 8;5 6 4 7];
h = patch('Faces',F,'Vertices',V);
set(h,'facealpha',0.05)
set(h,'facecolor',color)

end

