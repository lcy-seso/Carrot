clear all
close all

ColorSpec;
TestResults;

bs = sprintf('batch size = %d', batch_size);
marker_size = 6;
font_size = 16;
x = 1:8;

%%
gather_cpu = stack_cpu + reshape_cpu;
gather_cpu_dis = stack_cpu_dis + reshape_cpu_dis;

gather_cuda = stack_cuda + reshape_cuda;
gather_cuda_dis = stack_cuda_dis + reshape_cuda_dis;

scatter_cpu = unbind_cpu + view_cpu;
scatter_cpu_dis = unbind_cpu_dis + view_cpu_dis;

scatter_cuda = unbind_cuda + view_cuda;
scatter_cuda_dis = unbind_cuda_dis + view_cuda_dis;

%%
% Plot gather time on CPU and GPU
% for continuous and discontinuous data in memory

figure(1);
st = suptitle(bs);
set(st, 'FontSize', font_size, 'FontName', 'Kefa');

subplot(2, 2 ,1);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, gather_cpu,...
    '-o','MarkerSize', marker_size, 'Color', colorSpec{1}./255);
l2 = plot(x, gather_cpu_dis,...
    '-+','MarkerSize', marker_size, 'Color', colorSpec{2}./255);
l3 = plot(x, gather_cuda,...
    '-d','MarkerSize', marker_size, 'Color', colorSpec{5}./255);
l4 = plot(x, gather_cuda_dis,...
    '-*','MarkerSize', marker_size, 'Color', colorSpec{6}./255);

t = title('Gather Time (stack + reshape)');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

xticks(1:8);
xticklabels({16, 32, 64, 128, 256, 512, 1024, 2048});

xl = xlabel('hidden size');
set(xl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

yl = ylabel('time elapse (ms)');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3, l4],...
    'cpu-continuous data',...
    'cpu-discontinuous data',...
    'gpu-continuous data',...
    'gpu-discontinuous data',...
    'Location','northwest');
legend('boxoff');
lgd.FontSize = font_size;

%%
% Plot scatter time on CPU and GPU
% for continuous and discontinuous data in memory

subplot(2, 2 ,2);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, scatter_cpu,...
    '-o','MarkerSize', marker_size, 'Color', colorSpec{1}./255);
l2 = plot(x, scatter_cpu_dis,...
    '-+','MarkerSize', marker_size, 'Color', colorSpec{2}./255);
l3 = plot(x, scatter_cuda,...
    '-d','MarkerSize', marker_size, 'Color', colorSpec{5}./255);
l4 = plot(x, scatter_cuda_dis,...
    '-*','MarkerSize', marker_size, 'Color', colorSpec{6}./255);

t = title('Scatter Time (unbind + view)');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

xticks(1:8);
xticklabels({16, 32, 64, 128, 256, 512, 1024, 2048});

xl = xlabel('hidden size');
set(xl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

yl = ylabel('time elapse (ms)');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3, l4],...
    'cpu-continuous data',...
    'cpu-discontinuous data',...
    'gpu-continuous data',...
    'gpu-discontinuous data',...
    'Location','northwest');
legend('boxoff');
lgd.FontSize = font_size;

%%
% Plot computation time vs. gather time on CPU
% for continuous and discontinuous data in memory

subplot(2, 2, 3);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, lstm_cpu ./ gather_cpu,...
    '-o','MarkerSize',marker_size,'Color', colorSpec{1}./255);
l2 = plot(x, lstm_cpu_dis ./ gather_cpu_dis,...
    '-+','MarkerSize',marker_size,'Color', colorSpec{2}./255);

l3 = plot(x, lstm_cpu ./ scatter_cpu,...
    '-d','MarkerSize',marker_size,'Color', colorSpec{5}./255);
l4 = plot(x, lstm_cpu_dis ./ scatter_cpu_dis,...
    '-*','MarkerSize',marker_size,'Color', colorSpec{6}./255);

t = title('LSTMCell vs. Data Movement on CPU');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

xticks(1:8)
xticklabels({16, 32, 64, 128, 256, 512, 1024, 2048})
xl = xlabel('hidden size');
set(xl,'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

yl = ylabel('Ratio');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3, l4],...
    'lstm/gather-continuous-data',...
    'lstm/gather-discontinuous-data',...
    'lstm/scatter-continuous-data',...
    'lstm/scatter-discontinuous-data',...
    'Location','northwest');
legend('boxoff');
lgd.FontSize = font_size;

%%
% Plot computation time vs. gather time on CPU
% for continuous and discontinuous data in memory

subplot(2, 2, 4);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, lstm_cuda ./ gather_cuda,...
    '-o','MarkerSize', 5 ,'Color', colorSpec{1}./255);
l2 = plot(x, lstm_cuda_dis ./ gather_cuda_dis,...
    '-+','MarkerSize', 5 ,'Color', colorSpec{2}./255);
l3 = plot(x, lstm_cuda ./ scatter_cuda,...
    '-d','MarkerSize', 5 ,'Color', colorSpec{5}./255);
l4 = plot(x, lstm_cuda_dis ./ scatter_cuda_dis,...
    '-*','MarkerSize', 5 ,'Color', colorSpec{6}./255);

t = title('LSTMCell vs. Data Movement on GPU');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

xticks(1:8)
xticklabels({16, 32, 64, 128, 256, 512, 1024, 2048})
xl = xlabel('hidden size');
set(xl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

yl = ylabel('Ratio');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3, l4],...
    'lstm/gather-continuous-data',...
    'lstm/gather-discontinuous-data',...
    'lstm/scatter-continuous-data',...
    'lstm/scatter-discontinuous-data',...
    'Location','southeast');
legend('boxoff');
lgd.FontSize = font_size;
