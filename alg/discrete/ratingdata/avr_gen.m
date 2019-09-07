function mat = avr_gen(data)
    avg_score = sum(data,2) ./ sum(data>0,2);
    [M,N] = size(data);
    user_list = cell(M, 1);
    item_list = cell(M, 1);
    val_list = cell(M, 1);
    Rt = data';
    for i=1:M
        r = Rt(:,i);
        %idx = r > avg_score(i);
        [I,J,V] = find(r > avg_score(i));
        item_list{i} = I;
        val_list{i} = V;
        user_list{i} = i * J;
    end
    mat = sparse(cell2mat(user_list), cell2mat(item_list), cell2mat(val_list), M, N);
end