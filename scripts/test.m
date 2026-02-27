% load datafile
sess = 2;
sub = 54;
dir_path = fullfile("..", "gigaScienceDataset");
sess_str = sprintf('sess%02d', sess);
sub_str = sprintf('subj%02d', sub);
filename = sprintf('%s_%s_EEG_ERP.mat', sess_str, sub_str);
filepath = fullfile(dir_path, filename)
load(filepath)




baseDir = "../gigaScienceDataset";
sessStr = sprintf('sess%02d', 2);
subjStr = sprintf('subj%d', 54);
filename = sprintf('%s_%s_EEG_ERP.mat', sessStr, subjStr);

filepath = fullfile(baseDir, filename);
load(filepath)


% Define the parts separately
folder = '..';
datasetDir = 'gigaScienceDataset';
filename = 'sess02_subj54_EEG_ERP.mat';

% fullfile joins them with the correct slash for the current OS
filepath = fullfile(folder, datasetDir, filename);

load(filepath)
