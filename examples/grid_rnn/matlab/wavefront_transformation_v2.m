clc;
%clear all;
close all;



%% �ⲿ������ʾ��������ֱ�����������������Ʒ���
%ָ��x/y/z������ķ�Χ��step����Ϊ1����
figure()
view(3)
box on
hold on
grid on

x_value = 0:1:5;
y_value = 0:1:5;
z_value = 0:1:5;
plot_one_cube(x_value, y_value, z_value, 'blue');

x_value = 5:1:8;
y_value = 0:1:2;
z_value = 0:1:3;
plot_one_cube(x_value, y_value, z_value, 'green');

x_value = 8:1:12;
y_value = 0:1:4;
z_value = 0:1:6;
plot_one_cube(x_value, y_value, z_value, 'red');

%% �ⲿ������ʾ����һ���治��ֱ�����������������Ʒ���
figure()
view(3)
box on
hold on
grid on

x_value_bottom = 0:1:5;
x_value_upper = 3:1:8;
y_value = 0:1:4;
z_value = 0:1:3;
plot_one_rhombus_cube([x_value_bottom; x_value_upper], y_value, z_value, 'blue');
