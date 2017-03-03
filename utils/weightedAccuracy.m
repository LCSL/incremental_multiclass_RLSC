function wAcc = weightedAccuracy( Y, Ypred , classFreq )
%WEIGHTEDACCURACY works for +-1 coding
%   Detailed explanation goes here

    t = numel(classFreq);
    n = size(Y,1);
%     classCoeffs = ((1 - classFreq)/sum(1 - classFreq)) / min((1 - classFreq)/sum(1 - classFreq));
    classCoeffs = ((1 ./ classFreq) * max(classFreq));

    sampleClassIdx = cell(1,t);
    if t > 2
        for i = 1:t
            [ ~ , sampleClassIdx{i} ] = find(Y(i,:) == 1);
        end    
    elseif t == 2
        
        [ ~ , sampleClassIdx{1} ] = find(Y == 1);        
        [ ~ , sampleClassIdx{2} ] = find(Y ~= 1);        

        normFactor = sum(classCoeffs);

        C = bsxfun( @eq, Y, Ypred );
        
        % Reweight C
        F = zeros(size(C,1),size(C,2));
        for i = 1:t
            F(logical(sampleClassIdx{i})) = C(logical(sampleClassIdx{i}))*classCoeffs(i);
        end

        numCorrect = sum(F);
        wAcc = numCorrect / normFactor;   
    
    end
    
end
