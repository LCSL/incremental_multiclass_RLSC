addpath(genpath('utils'));
addpath(genpath('dataset'));
clearAllButBP;

% Path to results folder
resRoot = 'results';

datasetName = 'MNIST';

%% Set experimental results relative directory name

saveResult = 1;

customStr = '';
dt = clock;
dt = fix(dt); 	% Get timestamp
expDir = ['Exp_', customStr, '_' , mat2str(dt)];

resdir = [resRoot, '/', datasetName, '/', expDir];
mkdir(resdir);

%% Save current script in results
tmp = what;
[ST,I] = dbstack('-completenames');
copyfile([tmp.path ,'/', ST(1).name , '.m'],[ resdir ,'/', ST(1).name , '.m']);
copyfile([tmp.path ,'/', ST(2).name , '.m'],[ resdir ,'/', ST(2).name , '.m']);
copyfile([tmp.path ,'/dataConf_', datasetName , '_inc.m'],[ resdir ,'/dataConf_', datasetName , '_inc.m']);