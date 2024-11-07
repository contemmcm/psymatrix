clear all;
close all

psymatrix001_mean = load('results/nn2_004_accuracy_mean.txt');
psymatrix001_std = load('results/nn2_004_accuracy_std.txt');

ranking = load('../ranking.txt');

overall_bmodel = zeros(1,25);

model_idx = 1; model_label='openai-community/gpt2';
% model_idx = 2; model_label='google-bert/bert-base-multilingual-cased';
% model_idx = 3; model_label='google-bert/bert-base-cased';
% model_idx = 4; model_label='FacebookAI/roberta-base';
% model_idx = 5; model_label='FacebookAI/xlm-roberta-base';
% model_idx = 6; model_label='albert/albert-base-v2';
% model_idx = 7; model_label='xlnet/xlnet-base-cased';
% model_idx = 8; model_label='microsoft/mpnet-base';
% model_idx = 9; model_label='google/fnet-base';
% model_idx = 10; model_label='allenai/longformer-base-4096';
% model_idx = 11; model_label='studio-ousia/luke-base';
% model_idx = 12; model_label='studio-ousia/luke-japanese-base';
% model_idx = 13; model_label='bigscience/bloom-560m';
% model_idx = 14; model_label='bigscience/bloomz-560m';
% model_idx = 15; model_label='funnel-transformer/medium-base';
% model_idx = 16; model_label='Falconsai/question_answering';
% model_idx = 17; model_label='deepmind/language-perceiver';
% model_idx = 18; model_label='kssteven/ibert-roberta-base';
% model_idx = 19; model_label='uw-madison/nystromformer-1024';
% model_idx = 20; model_label='uw-madison/yoso-4096';
% model_idx = 21; model_label='flaubert/flaubert_base_cased';
% model_idx = 22; model_label='nghuyong/ernie-3.0-base-zh';
% model_idx = 23; model_label='facebook/opt-125m';
% model_idx = 24; model_label='facebook/opt-1.3b';


for i=1:24
    overall_bmodel(i+1) = length(find(ranking(model_idx,:)<=i)) / 146;
end
 


%psymatrix002_mean = load('results/nn2_064_accuracy_mean.txt');
%psymatrix002_std = load('results/nn2_064_accuracy_std.txt');

range = 0:(size(psymatrix001_mean,2));

x = range;

y001 = [0 mean(psymatrix001_mean,1)];
err001 = [0 mean(psymatrix001_std,1)];

%y002 = mean(psymatrix002_mean,1);
%err002 = mean(psymatrix002_std,1);

%y004 = mean(psymatrix004,1);
%err004 = std(psymatrix004,1);

rand_ranking = range ./ max(range);
figure;
hold on
plot(x,y001, '-o')
plot(x, overall_bmodel, '-s')
%plot(x,y002, '-s')

% plot(x,mean(psymatrix004,1), '-o')
% plot(x,mean(psymatrix10,1), '-o')

plot(range, rand_ranking, '--k')
patch([x flip(x)], [y001-err001 flip(y001+err001)], 'b', 'FaceAlpha',0.25, 'EdgeColor','none')

%patch([x flip(x)], [y002-err002 flip(y002+err002)], 'r', 'FaceAlpha',0.25, 'EdgeColor','none')
% patch([x flip(x)], [y004-err004 flip(y004+err004)], 'r', 'FaceAlpha',0.25, 'EdgeColor','none')
%hold on;
%errorbar(range, mean(psymatrix,1), std(psymatrix, 1))

hold off
xlabel('Top k')
ylabel('Optimal PTM Selection Accuracy')
grid on

legend('PsyMatrix', model_label, 'Random', 'Location', 'southeast')
axis tight