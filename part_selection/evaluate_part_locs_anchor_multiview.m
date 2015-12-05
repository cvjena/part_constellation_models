function [ channel_ids, part_visibility, anchor_points, shift_vectors, view_assignment, obj_value, err ] = ...
    evaluate_part_locs_anchor_multiview(part_locs, tr_ID, labels, no_selected_parts, no_visible_parts, view_count, iterations)    
    % Set no_visible_parts to NaN to avoid estimating visible parts

    %read part locations
    parts = array2table(part_locs,'VariableNames',{'Var1','Var2','Var3','Var4','Var5'});
%     parts = readtable('/home/simon/Datasets/CUB_200_2011/parts/part_locs.txt','Delimiter',' ','ReadVariableNames',false);
%     parts = readtable('/home/simon/Datasets/CUB_200_2011/parts/est_part_locs.txt','Delimiter',' ','ReadVariableNames',false);
    parts.Properties.VariableNames{'Var1'} = 'image';
    parts.Properties.VariableNames{'Var2'} = 'part';
    parts.Properties.VariableNames{'Var3'} = 'x';
    parts.Properties.VariableNames{'Var4'} = 'y';
    parts.Properties.VariableNames{'Var5'} = 'visible';
    part_ids = unique(parts.part);
    part_count = numel(part_ids);
    % Some temp variables
    parts_x = reshape(parts.x,numel(unique(parts.part)),numel(unique(parts.image)));
%     parts_x = parts_x(:,tr_ID);
    parts_y = reshape(parts.y,numel(unique(parts.part)),numel(unique(parts.image)));
%     parts_y = parts_y(:,tr_ID);
    part_locs = cat(3,parts_x,parts_y);
    part_locs = part_locs(1:part_count,:,:);
    part_ids = unique(parts.part);
    image_ids = unique(parts.image);
    part_count = numel(part_ids);
    image_count = numel(image_ids);
    
    % Load train test, perform selection only on train
    if nargin<2
        tr_ID = logical(load('/home/simon/Datasets/CUB_200_2011/tr_ID.txt'));
    end
    
    
    %% Constraints
    % Number of parts to select
    if nargin<3
        no_selected_parts = 5;
    end
    % Number of visible parts per image
    if nargin<4
        no_visible_parts = NaN;%ceil(no_selected_parts/2);
    end
    % Number of views
    if nargin<5
        view_count = 3;
    end


    part_visibility = nan(image_count,part_count);
    anchor_points = nan(image_count,2);
    shift_vectors = nan(numel(unique(labels)),part_count,view_count,2);
    view_assignment = false(image_count,view_count);
    model_errors = nan(image_count,1);
    fprintf('Working on class ');
    for c=unique(labels)'
        fprintf('%i ',c);
        class_tr_ID = tr_ID & labels==c;
        if sum(class_tr_ID)<1
            continue
        end
        best_obj_value = -Inf;
        for k=1:iterations 
            [ ~, h, a, d, s, obj_value, err ] = ...
                do_build_part_models(parts, part_locs, class_tr_ID,...
                no_selected_parts, no_visible_parts, view_count );
            if obj_value>best_obj_value
                best_obj_value = obj_value;
                best_h=h;
                best_a=a;
                best_d=d;
                best_s=s;
                best_err=err;
            end
        end
%             fprintf('%f\n',best_obj_value);
        part_visibility(class_tr_ID,:)=best_h;
        anchor_points(class_tr_ID,:) = best_a;
        shift_vectors(c,:,:,:)=best_d;
        view_assignment(class_tr_ID,:)=best_s;
        model_errors(class_tr_ID,:)=best_err;
    end
    
    % Inference for test images
    [~,channel_ids] = sort(-nansum(part_visibility,1));
%     channel_ids = channel_ids(1:no_selected_parts);
    % TODO: Here should be a proper inference
    part_visibility(~tr_ID,:)=false;
    part_visibility(~tr_ID,channel_ids)=true;
    part_visibility = logical(part_visibility);
end

function [ idx, part_visibility, anchor_points, shift_vectors, view_assignment, obj_value, err ] = do_build_part_models(parts, part_locs, tr_ID, no_selected_parts, no_visible_parts, view_count)    

    part_ids = unique(parts.part);
    image_ids = unique(parts.image);
    part_count = numel(part_ids);
    image_count = sum(tr_ID);
    
    part_locs = part_locs(:,tr_ID,:);

    %% Variables to estimate
    % View selection for each image
    s = false(image_count,view_count);
    % part selection b (indicator vector) for each view
    b = false(view_count,part_count);
    % Anchor points for each image
    a = zeros(image_count,2);
    % Shift vectors for each part in each view
    d = zeros(part_count, view_count,2);
    % Visibility of each part in each image
    h = false(image_count, part_count);
    
    %% Initialization 
    % Select a random view for each image
    for i=1:image_count
        s(i,randperm(view_count,1))=true;
    end
    % Select m random parts for each view
    for v=1:view_count
        b(v,randperm(part_count,no_selected_parts))=true;
    end
    % Set mean part position as default anchor point
    a = repmat(mean([parts.x(logical(parts.visible)) ...
        parts.y(logical(parts.visible))],1),size(a,1),1);
    % Set 0 as default shift vector
    d = zeros(size(d));
    if ~isnan(no_visible_parts)
        % Select no_visible_parts random parts for every view
        for i=1:image_count 
            % get the view for this image
            available_parts = find(b(s(i,:),:));
            h(i,available_parts(randperm(numel(available_parts),no_visible_parts)))=true;
        end
    end
    
    h = logical(reshape(parts.visible,numel(unique(parts.part)),numel(unique(parts.image))))';
    h = h(tr_ID,1:part_count);
    i = 0;
    
    done = false;
    best_obj_value = Inf;
    while ~done && ceil(i/2)<15
        i = i+1;
        old_b = b;
        
%         if mod(i,2)==1
%             fprintf('Running round %i \n',ceil(i/2));
%         end
        
        % General preparations
        % First, build a image_count x view_count x part_count x coordinates
        % Create singleton dimensions to fit target matrix shape
        % part_locs had shape part_count x image_count x coordinates
        mu_tmp = permute(part_locs,[2 4 1 3]);
        % a had shape image_count x coordinates
        a_tmp = permute(a,[1 3 4 2]);
        % d had shape part_count x view_count x coordinates
        d_tmp = permute(d,[4 2 1 3]);
        
        if mod(i,2)==1
            %% Estimate d
            % d has shape part_count x view_count x coordinates
            % Calculate d first, as we cannot do much wrong here (in contrast
            % to the part selection), and is required for the following steps
            % in order to produce any meaningful results
            % d is calculated by mean(mu-a along image_index)
            mu_a = bsxfun(@minus,mu_tmp,a_tmp);
            mu_a = repmat(mu_a,1,view_count,1,1);
            % Mask out data that is not visible
            mask = true(image_count,view_count,part_count);
            mask = bsxfun(@and, mask, permute(h,[1 3 2]));
            mask = repmat(mask,1,1,1,2);
            mu_a(~mask) = NaN;
            d = nanmean(mu_a,1);
            d = permute(d,[3 2 4 1]);

            %% Estimate a
            % a has shape image_count x coordinates
            % Calculate d first, as we cannot do much wrong here (in contrast
            % to the part selection), and is required for the following steps
            % in order to produce any meaningful results
            % d is calculated by mean(mu-a along image_index)
            mu_d = bsxfun(@minus,mu_tmp,d_tmp);
            mu_d(~mask) = NaN;
            a = nanmean(nanmean(mu_d,3),2);
            a = permute(a, [1 4 2 3]);
        else
            %% Preparations for b, h and s
            % calculate mu - (a + d) using bsxfun to automatically duplicate axis
            mu_a_d = bsxfun(@minus,mu_tmp,bsxfun(@plus,a_tmp,d_tmp));
            % Calculate the quadratic norm^2 along coordinate-axis
            mu_a_d = sum(mu_a_d.^2,4);

            %% Estimate h
            if ~isnan(no_visible_parts)
                % h has shape image_count x part_count
                est_h = false(size(h));
                % Shape of mu_a_d is (image_count x view_count x part_count)
                mask = true(image_count,view_count,part_count);
                mask = bsxfun(@and, mask, permute(s,[1 2 3]));
                mask = bsxfun(@and, mask, permute(b,[3 1 2]));
                mu_a_d_tmp = mu_a_d;
                mu_a_d_tmp(~mask) = NaN;
                mu_a_d_tmp = nansum(mu_a_d_tmp,2);
                % Only select parts to hide from chosen parts
                mu_a_d_tmp(mu_a_d_tmp == 0) = Inf;
                [~,idx] = sort(mu_a_d_tmp, 3);
                idx = permute(idx,[1 3 2]);
                idx = idx(:,1:no_visible_parts);
                idx2 = repmat((1:size(idx,1))',1,size(idx,2));
                est_h(sub2ind(size(h), idx2(:), idx(:))) = true;
            end

            %% Estimate s
            % s has shape image_count x view_count
            s = false(size(s));
            % Shape of mu_a_d is (image_count x view_count x part_count)
            mask = true(image_count,view_count,part_count);
            mask = bsxfun(@and, mask, permute(h,[1 3 2]));
            mask = bsxfun(@and, mask, permute(b,[3 1 2]));
            mu_a_d_tmp = mu_a_d;
            mu_a_d_tmp(~mask) = NaN;
            mu_a_d_tmp = nansum(mu_a_d_tmp,3);
            [~,idx] = sort(mu_a_d_tmp, 2);
            idx = idx(:,1);
            idx2 = repmat((1:size(idx,1))',1,size(idx,2));
            s(sub2ind(size(h), idx2(:), idx(:))) = true;

            %% Estimate b 
            % b has shape view_count x part_count
            b = false(size(b));
            % Shape of mu_a_d is (image_count x view_count x part_count)
            mask = true(image_count,view_count,part_count);
            mask = bsxfun(@and, mask, permute(s,[1 2 3]));
%             mask = bsxfun(@and, mask, permute(h,[1 3 2]));
            mu_a_d_tmp = mu_a_d;
            mu_a_d_tmp(~mask) = NaN;
            mu_a_d_tmp = nansum(mu_a_d_tmp,1);
            [~,idx] = sort(mu_a_d_tmp, 3);
            idx = permute(idx,[2 3 1]);
            idx = idx(:,1:no_selected_parts);
            idx2 = repmat((1:size(idx,1))',1,size(idx,2));
            b(sub2ind(size(b), idx2(:), idx(:))) = true;
%             % If you want to include distance between parts as selection
%             % criteria:
%             v = sum(pdist2(squeeze(d),squeeze(d)));
%             v = v/max(v(:))*10;
%             v = exp(v);
%             v = v/sum(v);
%             [~,idx] = sort(mu_a_d_tmp.*permute(v,[1 3 2]), 3);
            
            %% Remember old b to check for convergence
            if old_b == b
                done = true;
            else
                old_b = b;
            end
        end
        
        
%         % Calculate objective value            
%         mu_tmp = permute(part_locs,[2 4 1 3]);
%         % a had shape image_count x coordinates
%         a_tmp = permute(a,[1 3 4 2]);
%         % d had shape part_count x view_count x coordinates
%         d_tmp = permute(d,[4 2 1 3]);
%         % calculate mu - (a + d) using bsxfun to automatically duplicate axis
%         mu_a_d = bsxfun(@minus,mu_tmp,bsxfun(@plus,a_tmp,d_tmp));
%         % Calculate the quadratic norm^2 along coordinate-axis
%         mu_a_d = sum(mu_a_d.^2,4);
%         mask = true(image_count,view_count,part_count);
%         mask = bsxfun(@and, mask, permute(s,[1 2 3]));
%         mask = bsxfun(@and, mask, permute(h,[1 3 2]));
%         mask = bsxfun(@and, mask, permute(b,[3 1 2]));
%         mu_a_d_tmp = mu_a_d;
%         mu_a_d_tmp(~mask) = NaN;
%         new_obj_value = -nansum(mu_a_d_tmp(:));
% %         if true %new_obj_value < best_obj_value
% %             part_visibility = h;
% %             anchor_points = a;
% %             shift_vectors = d;
% %             best_obj_value = new_obj_value;
% %         end
%         fprintf('Objective value %10.0f\n', new_obj_value);
    end
%     channel_ids = idx(1:no_selected_parts);
%     save('part_selection_anchor_vgg19.mat','channel_ids');




    %% Get the error of each training image
    mu_tmp = permute(part_locs,[2 4 1 3]);
    % a had shape image_count x coordinates
    a_tmp = permute(a,[1 3 4 2]);
    % d had shape part_count x view_count x coordinates
    d_tmp = permute(d,[4 2 1 3]);
    % calculate mu - (a + d) using bsxfun to automatically duplicate axis
    mu_a_d = bsxfun(@minus,mu_tmp,bsxfun(@plus,a_tmp,d_tmp));
    % Calculate the quadratic norm^2 along coordinate-axis
    mu_a_d = sum(mu_a_d.^2,4);
    mask = true(image_count,view_count,part_count);
    mask = bsxfun(@and, mask, permute(s,[1 2 3]));
    mask = bsxfun(@and, mask, permute(h,[1 3 2]));
    mask = bsxfun(@and, mask, permute(b,[3 1 2]));
    mu_a_d(~mask) = NaN;
    err = nansum(nansum(mu_a_d,2),3);
    obj_value = -nansum(err);

    %% Return values 
    part_visibility = est_h;
    anchor_points = a;
    shift_vectors = d;
    view_assignment = s;
end