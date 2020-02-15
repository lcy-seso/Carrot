clear all
close all

ColorSpec;
TestResults;

bs = sprintf('batch size = %d', batch_size);
marker_size = 6;
font_size = 16;
x = 1:8;

%%
% Plot cat time on CPU and GPU
% for continuous and discontinuous data in memory

figure(1);
st = suptitle(bs);
set(st, 'FontSize', font_size, 'FontName', 'Kefa');

subplot(2, 2 ,1);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, cat_cpu,...
    '-o','MarkerSize', marker_size, 'Color', colorSpec{1}./255);
l2 = plot(x, cat_cpu_dis,...
    '-+','MarkerSize', marker_size, 'Color', colorSpec{2}./255);
l3 = plot(x, cat_cuda,...
    '-d','MarkerSize', marker_size, 'Color', colorSpec{5}./255);
l4 = plot(x, cat_cuda_dis,...
    '-*','MarkerSize', marker_size, 'Color', colorSpec{6}./255);

t = title('torch.cat Execution Time');
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
% Plot narrow time on CPU and GPU
% for continuous and discontinuous data in memory

subplot(2, 2 ,2);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, narrow_cpu,...
    '-o','MarkerSize', marker_size, 'Color', colorSpec{1}./255);
l2 = plot(x, narrow_cpu_dis,...
    '-+','MarkerSize', marker_size, 'Color', colorSpec{2}./255);
l3 = plot(x, narrow_cuda,...
    '-d','MarkerSize', marker_size, 'Color', colorSpec{5}./255);
l4 = plot(x, narrow_cuda_dis,...
    '-*','MarkerSize', marker_size, 'Color', colorSpec{6}./255);

t = title('torch.narrow Execution Time');
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
% Plot computation time vs. cat time on CPU
% for continuous and discontinuous data in memory

subplot(2, 2, 3);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, lstm_cpu ./ cat_cpu,...
    '-o','MarkerSize',marker_size,'Color', colorSpec{1}./255);
l2 = plot(x, lstm_cpu_dis ./ cat_cpu_dis,...
    '-+','MarkerSize',marker_size,'Color', colorSpec{2}./255);

l3 = plot(x, lstm_cpu ./ narrow_cpu,...
    '-d','MarkerSize',marker_size,'Color', colorSpec{5}./255);
l4 = plot(x, lstm_cpu_dis ./ narrow_cpu_dis,...
    '-*','MarkerSize',marker_size,'Color', colorSpec{6}./255);

t = title('LSTMCell vs. Data Movement on CPU');
set(t, 'FontSize', font_size, 'FontName', 'Kefa');

xticks(1:8)
xticklabels({16,32,64,128,256,512,1024,2048})
xl = xlabel('hidden size');
set(xl,'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

yl = ylabel('Ratio');
set(yl, 'FontSize', font_size, 'FontName', 'Kefa', 'Interpreter','none');

lgd = legend([l1, l2, l3, l4],...
    'lstm/cat-continuous-data',...
    'lstm/cat-discontinuous-data',...
    'lstm/narrow-continuous-data',...
    'lstm/narrow-discontinuous-data',...
    'Location','northwest');
legend('boxoff');
lgd.FontSize = font_size;

%%
% Plot computation time vs. cat time on CPU
% for continuous and discontinuous data in memory

subplot(2, 2, 4);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, lstm_cuda ./ cat_cuda,...
    '-o','MarkerSize', 5 ,'Color', colorSpec{1}./255);
l2 = plot(x, lstm_cuda_dis ./ cat_cuda_dis,...
    '-+','MarkerSize', 5 ,'Color', colorSpec{2}./255);
l3 = plot(x, lstm_cuda ./ narrow_cuda,...
    '-d','MarkerSize', 5 ,'Color', colorSpec{5}./255);
l4 = plot(x, lstm_cuda_dis ./ narrow_cuda_dis,...
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
    'lstm/cat-continuous-data',...
    'lstm/cat-discontinuous-data',...
    'lstm/narrow-continuous-data',...
    'lstm/narrow-discontinuous-data',...
    'Location','southeast');
legend('boxoff');
lgd.FontSize = font_size;


