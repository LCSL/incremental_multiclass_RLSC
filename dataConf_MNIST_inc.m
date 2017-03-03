dsRef = @MNIST;
coding = 'zeroOne';

ntr = 10000;
nte = [];

classes = 0:9; % classes to be extracted
imbClassArr = 1:10;   % Imbalanced class(es)

nLow = 1000;
lowFreq = 0.01;

if ~isempty(nLow)
    
    lowFreq = nLow/ntr;
end

highFreq = (1-lowFreq)/(numel(classes)-1);
trainClassFreq = [ highFreq * ones(1,9) lowFreq];
testClassFreq = [];

%% Alpha setting (only for recoding)

alphaArr = [0, 0.7];    % Array of the various recoding parameters 'alpha' to be tried.
                        % NOTE: alpha = 0 corresponds to naive RLSC (no
                        % recoding)
numAlpha = numel(alphaArr);
resultsArr = struct();
recod_alpha_idx  = 2;

%% Snapshot settings

snaps = 1:maxiter;   % Iterations for which incremental 
                    % solutions will be computed and compared
                    % on the test set in terms of accuracy
numSnaps = numel(snaps);
                    