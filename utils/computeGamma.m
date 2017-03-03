function [ gamma ] = computeGamma( p , currClassIdx )
%COMPUTEGAMMA computes the class reweighting factor of the given sample

    t = numel(p);

%     gamma = 1 / p(currClassIdx);
    gamma = prod( t * p([1:currClassIdx-1 , currClassIdx+1:t]));
%     gamma = prod(p([1:currClassIdx-1 , currClassIdx+1:t]));
end

