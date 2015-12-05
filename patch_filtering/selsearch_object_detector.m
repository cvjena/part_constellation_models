function [ output_args ] = selsearch_object_detector( channel_ids, part_locs, part_visibility, opts )
    output_file_train = opts.caffe_window_file_train;
    output_file_val = opts.caffe_window_file_val;
    output_file_bbox = opts.est_bbox_file;
      
    imagedir=opts.imagedir;
    imagelist_file = opts.imagelist_file;
    tr_ID_file = opts.tr_ID_file;
    labels_file = opts.labels_file;
    channels_for_boxes = channel_ids;
    
    add_est_bbox = opts.estimate_bbox;
    add_part_patches = true;
    add_proposals = true;
    write_bbox = opts.estimate_bbox;
    write_proposals = true;
    
    part_scales = opts.part_scales;
    scale_relative_to_bbox = false;    
    
    % Init caffe
    mean_file = opts.mean_mat_file;
    batch_size = opts.batch_size;
    crop_size = opts.crop_size;
    deploy = opts.deploy;
    model = opts.model;
    if write_bbox
    	matcaffe_init(1,deploy,model,1,0);
    end
    %read image list
    fid=fopen(imagelist_file,'r');
    imagelist=textscan(fid,'%s');
    imagelist=imagelist{1};
    fclose(fid);
    % load train test split
    tr_ID=logical(load(tr_ID_file));
%     tr_ID=true(size(imagelist,1),1);
    % Labels
    % Make sure labels start at 1
    labels=load(labels_file);
    labels=labels-min(labels(:))+1;
    
    
    %% Preparation starts
    parts = array2table(part_locs,'VariableNames',{'Var1','Var2','Var3','Var4','Var5'});
%     parts = readtable('/home/simon/Datasets/CUB_200_2011/parts/part_locs.txt','Delimiter',' ','ReadVariableNames',false);
%     parts = readtable('/home/simon/Datasets/CUB_200_2011/parts/est_part_locs.txt','Delimiter',' ','ReadVariableNames',false);
    parts.Properties.VariableNames{'Var1'} = 'image';
    parts.Properties.VariableNames{'Var2'} = 'part';
    parts.Properties.VariableNames{'Var3'} = 'x';
    parts.Properties.VariableNames{'Var4'} = 'y';
    parts.Properties.VariableNames{'Var5'} = 'visible';
    part_ids = unique(parts.part);
    image_ids = unique(parts.image);
    part_count = numel(part_ids);
    image_count = numel(image_ids);
    parts_x = reshape(parts.x,part_count,image_count);
    parts_y = reshape(parts.y,part_count,image_count);
    
    
    %% Calculation starts
    all_boxes = cell(image_count,1);
    all_images = cell(image_count,1);
    
    if write_proposals
        fid_train = fopen(output_file_train,'w');
        fid_val = fopen(output_file_val,'w');
        i_train = 0;
        i_test = 0;
        % Mean image
        mean_image = uint64(zeros(224,224,3));
        total_image_count = 0;
    end
    if write_bbox
        bbox_locs = nan(image_count,5);
    end
    
    fprintf('%s\n',datestr(now));
    for i=1:image_count%randperm(image_count)%
        if opts.verbose_output
            fprintf('Working on %i: %s\n',i,imagelist{i});
        end
%         if tr_ID(i)
%             continue
%         end

        im = imread([imagedir '/' imagelist{i}]);
        if size(im,3)==1
            im=repmat(im,1,1,3);
        end
        %% Get the propsals for the image 
        if add_proposals
    %         all_images{i} = im;
            colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
            colorType = colorTypes{1:5}; 
            % Here you specify which similarity functions to use in merging
            simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
            simFunctionHandles = simFunctionHandles(1:4); % Two different merging strategies
            % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
            % Note that by default, we set minSize = k, and sigma = 0.8.
            k = 200; % controls size of segments of initial segmentation. 
            minSize = k;
            sigma = 0.8;
            % Selective search start
            [all_boxes{i}] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
            all_boxes{i} = BoxRemoveDuplicates(all_boxes{i});
        else
            all_boxes{i} = [];
        end
        boxes = all_boxes{i};

        %% Filter out boxes with zero size or which are too narrow
        if size(boxes,1)<1
            % Always keep the whole image (in case sel search fails)
            boxes = [1 1 size(im,1) size(im,2)];
        else
            box_size_selection = (boxes(:,3)-boxes(:,1)).*(boxes(:,4)-boxes(:,2))>0 & ...
                (boxes(:,3)-boxes(:,1))>30 & (boxes(:,4)-boxes(:,2))>30;
            boxes = boxes(box_size_selection,:);
        end
        
        %% Transform part locs to actual part locations in the image
        if add_proposals
            % Get all relevant and visible part positions
            channel_ids = find(part_visibility(i,:));
            selection = parts.visible((i-1)*part_count + channel_ids);
            cur_channels = channel_ids(logical(selection));
            cur_locs = [parts_x(cur_channels,i) parts_y(cur_channels,i)];

            for k=1:size(cur_locs,1)
                x=cur_locs(k,1);
                y=cur_locs(k,2);
                % calc ratio 
                ratio_x = opts.crop_size / size(im,2);
                ratio_y = opts.crop_size / size(im,1);
                cur_locs(k,1)=int32(x/ratio_x);
                cur_locs(k,2)=int32(y/ratio_y);
            end
    %             hold off
        else
            cur_locs = [];
        end

        %% Add part based boxes
        if add_part_patches
            box_part_selection = false(size(boxes,1),1);
            for part_scale = part_scales
                % Get visibile parts for the custom channel selection
                part_based_locs = [parts_x(channels_for_boxes,i) parts_y(channels_for_boxes,i)];
                if scale_relative_to_bbox
                    box_size = 0.5*sqrt(cur_box(4)*cur_box(5));
                else
                    box_size = part_scale*sqrt(size(im,1)*size(im,2));
                end
                for c=1:size(part_based_locs,1)
                    if parts.visible((i-1)*part_count + channels_for_boxes(c))
                        x=part_based_locs(c,1);
                        y=part_based_locs(c,2);
                        if i>image_count
                            x=opts.crop_size-x;
                        end
                        ratio_x = opts.crop_size / size(im,2);
                        ratio_y = opts.crop_size / size(im,1);
                        x=int32(x/ratio_x);
                        y=int32(y/ratio_y);
                        x_min = max(x-box_size/2, 1);
                        x_max = min(x+box_size/2, size(im,2));
                        y_min = max(y-box_size/2, 1);
                        y_max = min(y+box_size/2, size(im,1));
                        boxes = [boxes;y_min x_min y_max x_max];
                        box_part_selection = [box_part_selection;true];
                    end
                end
            end
        end

        %% Bounding box estimation
        if write_bbox
            % Classify all boxes        
            batch_data = {};
            for b=1:size(boxes,1)
                batch_data = [batch_data; im(boxes(b,1):boxes(b,3),boxes(b,2):boxes(b,4),:)];
            end
            probs = caffe_features(batch_data,'prob',mean_file,batch_size,crop_size);
            if tr_ID(i)
                pred_class = labels(i,:);
            else
                % Predict the class 
                % Use the most confident classification result as class
                % pred
                [val,pred_class]=max(max(probs(:,2:end),[],1),[],2);
            end
            % Take the bbox with the most sure classification
            [~,idx ] = sort(-probs(:,pred_class+1));
            bbox_locs(i,:) = [i, ...
                boxes(idx(1),2)                ,boxes(idx(1),1),...
                boxes(idx(1),4)-boxes(idx(1),2),boxes(idx(1),3)-boxes(idx(1),1)];
        end

        %% Proposals
        if write_proposals 
            if add_proposals
                %% Decide foreground and background boxes according to part location
                % Count how many parts are inside the proposed box
                fg_bg_selection = zeros(size(boxes,1),1);
                cur_boxes = [];
                for loc = cur_locs'
                    % loc has shape [x=col y=row]
                    % Check which boxes contain this part and count 
                    fg_bg_selection = fg_bg_selection + ...
                        (boxes(:,1)<=loc(2) & boxes(:,3)>=loc(2)& ...
                        boxes(:,2)<=loc(1) & boxes(:,4)>=loc(1));
                end
                % Take only boxes with three or more part detections
                fg_bg_selection=fg_bg_selection>numel(cur_channels)-3;

                box_selection = fg_bg_selection | box_part_selection;
                % Always take the full image
                box_selection(1) = true;
            elseif add_part_patches
                box_selection = box_part_selection;
            else
                box_selection = [];
            end
            % Add the estimated bounding box
            if add_est_bbox
                boxes = [boxes;[bbox_locs(i,3),bbox_locs(i,2),...
                    bbox_locs(i,5)+bbox_locs(i,3),bbox_locs(i,4)+bbox_locs(i,2)]];
                box_selection = [box_selection;true];
            end
            
            if opts.verbose_output
                fprintf('Found %i boxes\n',sum(box_selection));
            end

            %% Now store these bboxes in text file
            if tr_ID(i,:)
                fid = fid_train;
                fprintf(fid,'# %i\n',i_train); % Image id
                i_train = i_train + 1;
            else
                fid = fid_val;
                fprintf(fid,'# %i\n',i_test); % Image id
                i_test = i_test + 1;
            end
            fprintf(fid,'%s\n',[imagedir imagelist{i}]); % absolute image path
            fprintf(fid,'%i\n',3); % num channels
            fprintf(fid,'%i\n',size(im,1)); % height
            fprintf(fid,'%i\n',size(im,2)); % width 
            fprintf(fid,'%i\n',size(boxes,1)); % num_windows
            for b = 1:size(boxes,1)
                fprintf(fid,'%i %i %.0f %.0f %.0f %.0f\n',labels(i,:),box_selection(b,:),...
                    boxes(b,2),boxes(b,1),boxes(b,4),boxes(b,3));
                if opts.calculate_mean
                    mean_image = mean_image + uint64(imresize(im(boxes(b,1):boxes(b,3),boxes(b,2):boxes(b,4),:),[224 224]));
                    total_image_count = total_image_count + 1;
                end
%                 if box_selection(b,:)
%                     imshow(im(boxes(b,1):boxes(b,3),boxes(b,2):boxes(b,4),:));
%                     waitforbuttonpress; clf
%                 end
            end
        end
    end
    if write_proposals
        fclose(fid_train);
        fclose(fid_val);
        if opts.calculate_mean
            mean_image = double(mean_image/total_image_count);
            save('tmp_mean.mat','mean_image');
        end
    end
    if write_bbox
        dlmwrite(output_file_bbox,bbox_locs,'Delimiter',' ');
    end
end

