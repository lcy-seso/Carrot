clear all
close all

%%
colorSpec{1}= [220 20 60]; % red
colorSpec{2}= [255 99 71];

colorSpec{3} = [100 149 237]; % blue
colorSpec{4} = [0 0 139];

colorSpec{5} = [139 69 19];  % brown
colorSpec{6} = [205 133 63];

font_size = 16;
marker_size = 10;
width = 1.5;
%%
x = 1:4; % four implementations

cpu_bs16 = [165.5627, 16.9245, 17.9632, 12.8334];
gpu_bs16 = [371.8439, 21.9785, 22.6913, 13.3294];

cpu_bs32 = [334.7863, 36.1208, 35.8145, 27.0814];
gpu_bs32 = [732.4716, 45.8285, 46.1304, 27.0525];

cpu_bs64 = [689.3593, 66.0319, 68.2955, 54.2287];
gpu_bs64 = [92.1622, 92.7896, 54.4703]; % naive implementation is OOM.

%%
% Compare different implementations of GridLSTM.

st = suptitle('Compare four implementations of GridLSTM');
set(st, 'FontSize', font_size, 'FontName', 'Kefa');

subplot(1,3,1);
t = title('CPU and GPU execution');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

box on;
grid on;
hold on;
axis tight;

l1 = plot(x, cpu_bs16,...
    '-o','MarkerSize', marker_size,...
    'Color', colorSpec{1}./255,...
    'MarkerFaceColor', colorSpec{1}./255);
l1.LineWidth = width;

l2 = plot(x, gpu_bs16,...
    '-^','MarkerSize', marker_size,...
    'Color', colorSpec{2}./255,...
    'MarkerFaceColor', colorSpec{2}./255);
l2.LineWidth = width;

l3 = plot(x, cpu_bs32,...
    '-d','MarkerSize', marker_size,...
    'Color', colorSpec{3}./255,...
    'MarkerFaceColor', colorSpec{3}./255);
l3.LineWidth = width;

l4 = plot(x, gpu_bs32,...
    '-*','MarkerSize', marker_size,...
    'Color', colorSpec{4}./255,...
    'MarkerFaceColor', colorSpec{4}./255);
l4.LineWidth = width;

l5 = plot(x, cpu_bs64,...
    '-x','MarkerSize', marker_size,...
    'Color', colorSpec{5}./255,...
    'MarkerFaceColor', colorSpec{5}./255);
l5.LineWidth = width;

l6 = plot(2:4, gpu_bs64,...
    '-s','MarkerSize', marker_size,...
    'Color', colorSpec{6}./255,...
    'MarkerFaceColor', colorSpec{6}./255);
l6.LineWidth = width;

xl = xlabel('different implementations');
set(xl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');
xticks(1:4);
xticklabels({'V0','V1','V2','V3'});

yl = ylabel('time elapse (s)');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3, l4, l5, l6],...
    'cpu-batch\_size-16',...
    'gpu-batch\_size-16',...
    'cpu-batch\_size-32',...
    'gpu-batch\_size-32',...
    'cpu-batch\_size-64',...
    'gpu-batch\_size-64',...
    'Location','northeast');
legend('boxoff');
lgd.FontSize = font_size;

%%

subplot(1,3,2);

t = title('CPU execution');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

box on;
grid on;
hold on;
axis tight;

l1 = plot(x, cpu_bs16,...
    '-o','MarkerSize', marker_size,...
    'Color', colorSpec{1}./255,...
    'MarkerFaceColor', colorSpec{1}./255);
l1.LineWidth = width;

l2 = plot(x, cpu_bs32,...
    '-d','MarkerSize', marker_size,...
    'Color', colorSpec{3}./255,...
    'MarkerFaceColor', colorSpec{3}./255);
l2.LineWidth = width;

l3 = plot(x, cpu_bs64,...
    '-x','MarkerSize', marker_size,...
    'Color', colorSpec{5}./255,...
    'MarkerFaceColor', colorSpec{5}./255);
l3.LineWidth = width;

xl = xlabel('different implementations');
set(xl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');
xticks(1:4);
xticklabels({'V0','V1','V2','V3'});

yl = ylabel('time elapse (s)');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3],...
    'batch\_size-16',...
    'batch\_size-16',...
    'batch\_size-32',...
    'Location','northeast');
legend('boxoff');
lgd.FontSize = font_size;

%%
% Compare 4 GPU implementations.

subplot(1,3,3);

t = title('GPU execution');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

box on;
grid on;
hold on;
axis tight;

l1 = plot(x, gpu_bs16,...
    '-^','MarkerSize', marker_size,...
    'Color', colorSpec{2}./255,...
    'MarkerFaceColor', colorSpec{2}./255);
l1.LineWidth = width;

l2 = plot(x, gpu_bs32,...
    '-*','MarkerSize', marker_size,...
    'Color', colorSpec{4}./255,...
    'MarkerFaceColor', colorSpec{4}./255);
l2.LineWidth = width;

l3 = plot(2:4, gpu_bs64,...
    '-s','MarkerSize', marker_size,...
    'Color', colorSpec{6}./255,...
    'MarkerFaceColor', colorSpec{6}./255);
l3.LineWidth = width;

xl = xlabel('different implementations');
set(xl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');
xticks(1:4);
xticklabels({'V0','V1','V2','V3'});

yl = ylabel('time elapse (s)');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3],...
    'batch\_size-16',...
    'batch\_size-16',...
    'batch\_size-32',...
    'Location','northeast');
legend('boxoff');
lgd.FontSize = font_size;