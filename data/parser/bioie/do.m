d = dir('source*');
T = randperm(length(d));
for i=1:length(d)
    gs = d(i).name;
    m = regexp(gs,'\d');
    if isempty(m)
        continue;
    end
    t = sprintf('%s%04d%s',gs(1:min(m)-1), T(i), gs(max(m)+1:end));
    system(['mv ' gs ' ' t]);
end
