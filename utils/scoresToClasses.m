% Compute predictions matrix from real-valued scores matrix
function Ypred = scoresToClasses( Yscores , outputFormat )    

T = size(Yscores,2);

if T == 2
    if strcmp(outputFormat, 'zeroOne')
        Ypred = zeros(size(Yscores));
    elseif strcmp(outputFormat, 'plusMinusOne')
        Ypred = -1 * ones(size(Yscores));
    elseif strcmp(outputFormat, 'plusOneMinusBalanced')
        Ypred = -1/(T - 1) * ones(size(Yscores));
    end
    Ypred(Yscores > 0) = 1;
else
    if strcmp(outputFormat, 'zeroOne')
        Ypred = zeros(size(Yscores));
    elseif strcmp(outputFormat, 'plusMinusOne')
        Ypred = -1 * ones(size(Yscores));
    elseif strcmp(outputFormat, 'plusOneMinusBalanced')
        Ypred = -1/(T - 1) * ones(size(Yscores));
    end

%                 for i = 1:size(Ypred,1)
%                     [~,maxIdx] = max(Yscores(i,:));
%                     Ypred(i,maxIdx) = 1;
%                 end
    [~,maxIdx] = max(Yscores , [] , 2);
    indices = sub2ind(size(Ypred), 1:numel(maxIdx), maxIdx'); % To
%                 be removed
%                 indices = 1:numel(maxIdx) + (maxIdx'-1)*size(Ypred,1);
    Ypred(indices) = 1;
end
end