clear all;
close all;
%%
colorSpec{1}= [	25 25 112];
colorSpec{2}= [16 78 139];

colorSpec{3} = [0 205 102];
colorSpec{4} = [84 139 84];

colorSpec{5} = [139 28 98];
colorSpec{6} = [205 41 144];

colorSpec{7}= [255 165 0];
colorSpec{8}= [255 140 0];

colorSpec{9} = [123 104 238];
colorSpec{10} = [147 112 219];

colorSpec{11} = [30 144 255];
colorSpec{12} = [0 191 255];

colorSpec{13}= [139 69 19];
colorSpec{14}= [160 82 45];

%% continuous memory
% 
% batch_size = 64;
% stack_cpu = [0.094962,0.081134,0.083971,0.081611,0.082612,0.090957,0.130057,0.184584];
% lstm_cpu = [3.979659,4.785204,4.257512,4.515433,5.183268,7.197952,23.938441,126.401353];
% unbind_cpu = [0.150323,0.108528,0.094223,0.093198,0.088978,0.093150,0.107360,0.107408];
% 
% stack_cuda = [0.200272,0.118136,0.118327,0.117135,0.118232,0.120258,0.139689,0.138712];
% lstm_cuda = [0.794291,0.769019,0.774908,0.767398,0.766349,0.809526,0.874996,0.895524];
% unbind_cuda = [0.083303,0.080299,0.080585,0.080872,0.080919,0.082302,0.089073,0.088310];
% 
% % discontinuous memory
% dis_stack_cpu = [0.145459,0.159860,0.110507,0.162125,0.099635,0.113583,0.145364,0.180173];
% dis_lstm_cpu = [4.077172,4.911566,4.427481,4.620647,5.156016,7.887912,31.950188,121.242261];
% dis_unbind_cpu = [0.118613,0.126791,0.114894,0.130606,0.097561,0.101376,0.133801,0.121379];
% 
% dis_stack_cuda = [0.207210,0.241256,0.195026,0.172305,0.144267,0.145102,0.157666,0.158787];
% dis_lstm_cuda = [0.820065,1.510000,1.192307,1.020455,0.862885,0.882363,0.907087,0.929165];
% dis_unbind_cuda = [0.087142,0.169039,0.131154,0.112486,0.096154,0.097799,0.104165,0.105929];
%%

% batch_size = 256;
% stack_cpu = [0.232530,0.235510,0.240779,0.248122,0.257373,0.291491,0.395727,0.497913];
% lstm_cpu = [14.121294,14.329195,15.313292,16.553140,19.063640,26.597214,83.702493,462.047338];
% unbind_cpu = [0.269008,0.269938,0.272703,0.281954,0.287247,0.309801,0.319672,0.306511];
% 
% stack_cuda = [0.320482,0.322723,0.318050,0.320625,0.316882,0.335503,0.338197,0.354981];
% lstm_cuda = [2.553654,2.528763,2.519321,2.506185,2.519798,2.582407,2.632642,2.592278];
% unbind_cuda = [0.272226,0.273728,0.272250,0.284362,0.272250,0.281262,0.292325,0.290608];
% 
% dis_stack_cpu = [0.273800,0.289488,0.291872,0.313950,0.355172,0.410581,0.460696,0.606441];
% dis_lstm_cpu = [14.520311,14.716077,15.771627,16.957927,20.111799,28.284955,81.416988,442.170358];
% dis_unbind_cpu = [0.374436,0.383997,0.386310,0.382924,0.429869,0.454235,0.464368,0.455165];
% 
% dis_stack_cuda = [0.379467,0.342846,0.352240,0.345898,0.350118,0.348306,0.359154,0.358391];
% dis_lstm_cuda = [2.620363,2.582669,2.591515,2.575374,2.573323,2.607799,2.605677,4.868555];
% dis_unbind_cuda = [1.004171,0.351644,0.344372,0.344372,0.335932,0.352764,0.344563,0.353813];
%%
% batch_size = 512;
% stack_cpu = [0.444937,0.457740,0.453138,0.467539,0.494337,0.591016,0.764418,1.058960];
% lstm_cpu = [27.808809,28.155184,30.338788,33.055925,38.756299,56.816697,181.598377,923.092961];
% unbind_cpu = [0.590801,0.518894,0.518155,0.544500,0.550389,0.588179,0.573611,0.560904];
% 
% stack_cuda = [0.590038,0.594974,0.592375,0.597644,0.592732,0.607562,0.607324,0.616646];
% lstm_cuda = [4.888177,4.836226,4.781675,4.784513,4.770851,4.923391,4.854846,4.923010];
% unbind_cuda = [0.520277,0.527120,0.530720,0.534439,0.523043,0.532317,0.565529,0.548315];
% 
% dis_stack_cpu = [0.535440,0.568819,0.574803,0.615215,0.689006,0.804782,0.927377,1.260018];
% dis_lstm_cpu = [28.207850,29.004216,30.980849,33.706164,39.056325,53.846884,158.583808,879.048777];
% dis_unbind_cpu = [0.675583,0.704384,0.729942,0.762153,0.800896,0.838017,0.832129,0.829387];
% 
% dis_stack_cuda = [0.708604,0.664091,0.662160,0.664020,0.666213,0.663233,0.659966,0.661302];
% dis_lstm_cuda = [4.955792,4.925489,4.909921,4.923558,4.950333,5.050039,15.995502,57.133126];
% dis_unbind_cuda = [3.491306,0.660825,0.640607,0.655007,0.642729,0.667000,0.658751,0.662065];

%% 
batch_size = 1024;
stack_cpu = [0.871444,0.888801,0.897169,0.954270,1.032734,1.213455,1.437449,2.097440];
lstm_cpu = [56.105661,57.032394,61.739826,67.921090,83.866858,131.884766,427.446008,1995.865989];
unbind_cpu = [1.083398,1.909876,1.124907,1.179028,1.164460,1.238751,1.146793,1.177001];

stack_cuda = [1.142621,1.145053,1.135063,1.153159,1.152563,1.155114,1.162767,1.179314];
lstm_cuda = [9.425688,9.456444,9.345913,9.382534,9.349704,9.596586,24.184394,70.940375];
unbind_cuda = [1.068854,1.067400,1.073003,1.064038,1.069999,1.076794,1.765060,1.080108];


dis_stack_cpu = [1.114893,1.180029,1.181817,1.253843,1.382470,1.629305,2.021813,2.592087];
dis_lstm_cpu = [57.009268,58.732700,62.976193,67.957354,79.198337,110.162616,317.258811,1764.874101];
dis_unbind_cpu = [1.495934,1.584077,1.571035,1.620221,1.602054,1.624537,1.631546,2.334642];

dis_stack_cuda = [1.886272,1.329517,1.304960,1.296902,1.296830,1.323509,1.298952,1.308823];
dis_lstm_cuda = [9.574437,9.665298,9.529614,9.654593,9.560251,9.843135,32.282138,120.553565];
dis_unbind_cuda = [1.347780,1.379561,1.332068,1.383185,1.339054,1.380706,1.318216,1.359248];

%%
bs = sprintf('batch size = %d', batch_size);
marker_size = 6;

x = 1:8;
%%
subplot(2,2,1);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, stack_cpu,...
    '-o','MarkerSize',marker_size,'Color', colorSpec{1}./255);
l2 = plot(x, dis_stack_cpu,...
    '-+','MarkerSize',marker_size,'Color', colorSpec{2}./255);
l3 = plot(x, stack_cuda,...
    '-d','MarkerSize',marker_size,'Color', colorSpec{5}./255);
l4 = plot(x, dis_stack_cuda,...
    '-*','MarkerSize',marker_size,'Color', colorSpec{6}./255);

title({'torch.stack';bs})
xl = xlabel('hidden size');

xticks(1:8)
xticklabels({16,32,64,128,256,512,1024,2048})

set(xl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
yl = ylabel('time elapse (ms)');
set(yl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
legend([l1, l2, l3, l4], 'CPU-continuous', 'CPU-discontinuous',...
    'GPU-continuous', 'GPU-discontinuous');

%%
subplot(2,2,2);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, unbind_cpu,...
    '-o','MarkerSize',marker_size,'Color', colorSpec{1}./255);
l2 = plot(x, dis_unbind_cpu,...
    '-+','MarkerSize',marker_size,'Color', colorSpec{2}./255);
l3 = plot(x, unbind_cuda,...
    '-d','MarkerSize',marker_size,'Color', colorSpec{5}./255);
l4 = plot(x, dis_unbind_cuda,...
    '-*','MarkerSize',marker_size,'Color', colorSpec{6}./255);

title({'torch.unbind';bs})
xl = xlabel('hidden size');

xticks(1:8)
xticklabels({16,32,64,128,256,512,1024,2048})

set(xl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
yl = ylabel('time elapse (ms)');
set(yl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
legend([l1, l2, l3, l4], 'CPU-continuous', 'CPU-discontinuous',...
    'GPU-continuous', 'GPU-discontinuous');

%% 
subplot(2,2,3);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, lstm_cpu./stack_cpu,...
    '-o','MarkerSize',marker_size,'Color', colorSpec{1}./255);
l2 = plot(x, dis_lstm_cpu./dis_stack_cpu,...
    '-+','MarkerSize',marker_size,'Color', colorSpec{2}./255);

l3 = plot(x, lstm_cpu./unbind_cpu,...
    '-d','MarkerSize',marker_size,'Color', colorSpec{5}./255);
l4 = plot(x, dis_lstm_cpu./dis_unbind_cpu,...
    '-*','MarkerSize',marker_size,'Color', colorSpec{6}./255);

title({'LSTMCell vs. Data Movement on CPU';bs})
xl = xlabel('hidden size');

xticks(1:8)
xticklabels({16,32,64,128,256,512,1024,2048})

set(xl,'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
yl = ylabel('Ratio');
set(yl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
legend([l1, l2, l3, l4], 'lstm/stack-continuous', 'lstm/stack-discontinuous',...
    'lstm/unbind-continuous', 'lstm_cuda/unbind-discontinuous');

%%
subplot(2,2,4);
box on;
grid on;
hold on;
axis tight;

l1 = plot(x, lstm_cuda./stack_cuda,...
    '-o','MarkerSize',5,'Color', colorSpec{1}./255);
l2 = plot(x, dis_lstm_cuda./dis_stack_cuda,...
    '-+','MarkerSize',5,'Color', colorSpec{2}./255);
l3 = plot(x, lstm_cuda./unbind_cuda,...
    '-d','MarkerSize',5,'Color', colorSpec{5}./255);
l4 = plot(x, dis_lstm_cuda./dis_unbind_cuda,...
    '-*','MarkerSize',5,'Color', colorSpec{6}./255);

xticks(1:8)
xticklabels({16,32,64,128,256,512,1024,2048})
title({'LSTMCell vs. Data Movement on GPU';bs})
xl = xlabel('hidden size');
set(xl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
yl = ylabel('Ratio');
set(yl, 'FontSize', 16, 'FontName', 'Kefa', 'Interpreter','none');
legend([l1, l2, l3, l4], 'lstm/stack-continuous', 'lstm/stack-discontinuous',...
    'lstm/unbind-continuous', 'lstm/unbind-discontinuous');
