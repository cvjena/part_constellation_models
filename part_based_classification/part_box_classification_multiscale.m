function [ output_args ] = part_box_classification_multiscale( channel_ids, part_locs, opts )
   %% Datasets
   imagedir=opts.imagedir;
   imagelist_file = opts.imagelist_file;
   tr_ID_file = opts.tr_ID_file;
   labels_file = opts.labels_file;
   bbox_file = opts.est_bbox_file;

    %% Params
    layer_parts = opts.feature_part;
    layer_image = opts.feature_global;
    use_flipped = opts.use_flipped;
    use_bounding_box = opts.use_bounding_box;
    pyramid_levels = opts.pyramid_levels;
    use_parts = opts.use_parts;
    rand_tr_images = opts.rand_tr_images;
    rand_tr_part = opts.rand_tr_part;
    part_scales = opts.part_scales;%0.31;%
    params=opts.svm_params;
    
    scale_relative_to_bbox = false;

    parfor_workers = opts.parfor_workers;
    use_parfor = opts.use_parfor;
    
    mean_file = opts.mean_mat_file;
    batch_size = opts.batch_size;
    crop_size = opts.crop_size;
    deploy = opts.deploy;
    model = opts.model;

    rng('shuffle');
    %read image list
    fid=fopen(imagelist_file,'r');
    imagelist=textscan(fid,'%s');
    imagelist=imagelist{1};
    fclose(fid);
    % Labels
    labels=load(labels_file);
    % load train test split
    if rand_tr_images>0
        [ tr_ID ] = createTrainTest( labels, rand_tr_images, opts.rand_tr_part );
    else
        train_test=logical(load(tr_ID_file));
        tr_ID = train_test;%(:,2);
    end
    % Bounding boxes 
    bboxes = [];
    if use_bounding_box
    %     bboxes = load([basedir '/bounding_boxes.txt'])+1;
        bboxes = load(bbox_file);
    end
    
    if use_parts
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
    else
        image_count = size(imagelist,1);
        channel_ids = [];
        parts = [];
        part_count = [];
        parts_x = [];
        parts_y = [];
    end
    
    if use_flipped
        flipped_image_count = image_count + sum(tr_ID);
        labels = [labels;labels(tr_ID)];
        image_idx = [(1:image_count)';find(tr_ID)];
        tr_ID = [tr_ID;true(sum(tr_ID),1)];
    else
        flipped_image_count = image_count;
        image_idx = (1:image_count)';
    end
    
    matcaffe_init(1,deploy,model,1,0);
    if opts.use_parts
        f = caffe_features({[0]},layer_parts,mean_file,batch_size,crop_size);
    else
        f = [];
    end
    f2 = caffe_features({[0]},layer_image,mean_file,batch_size,crop_size);
    caffe('reset');
    
    if use_parfor
        if ~isempty(gcp('nocreate'))%matlabpool('size')
            pctRunOnAll caffe('reset')
        else
            caffe('reset');
            parpool(parfor_workers);
        end
    end
    parfor (i=1:parfor_workers, opts.parfor_arg)
        matcaffe_init(1,deploy,model,1,mod(i,opts.gpu_count));
    end
    
    num_patches_per_image = 0;
    if use_parts
        num_patches_per_image = num_patches_per_image + numel(part_scales)*numel(channel_ids);
    end
    if use_bounding_box
        num_patches_per_image = num_patches_per_image+1;
    end
    if pyramid_levels>0
        num_patches_per_image = num_patches_per_image+(1-4^(pyramid_levels+1))/-3 - 1;
    end
    
    features = sparse(flipped_image_count, num_patches_per_image*size(f,2)+size(f2,2));
%     visible = true(flipped_image_count, numel(channel_ids));
    feature_count = size(features,2);
    parfor (i=1:flipped_image_count, opts.parfor_arg) % randperm(image_count)%[1:10 11788+(1:10)]%
        cur_image_idx = image_idx(i);
        if opts.verbose_output
            fprintf('Working on %i: %s\n',i,imagelist{cur_image_idx});
        end
        im = imread([imagedir '/' imagelist{cur_image_idx}]);
        if i>image_count
            im = flip(im,2);
        end
        batch_data = {};
        missing_data = false(0,0);
        
        if use_bounding_box
            cur_box = bboxes(cur_image_idx,:);
            cur_box(4) = min(cur_box(4),size(im,2)-cur_box(2)+1);
            cur_box(5) = min(cur_box(5),size(im,1)-cur_box(3)+1);
            batch_data = [batch_data;im(cur_box(3):(cur_box(3)+cur_box(5)-1),cur_box(2):(cur_box(2)+cur_box(4)-1),:)];
            missing_data = [missing_data;false];
%              box_size = 0.5*sqrt(cur_box(4)*cur_box(5));
        end 
        
        if use_parts
            for part_scale = part_scales
                % Get all relevant and visible part positions
                selection = parts.visible((cur_image_idx-1)*part_count + channel_ids);
                visible_channels = channel_ids(logical(selection));
                cur_locs = [parts_x(channel_ids,cur_image_idx) parts_y(channel_ids,cur_image_idx)];
                if scale_relative_to_bbox
                    box_size = 0.5*sqrt(cur_box(4)*cur_box(5));
                else
                    box_size = part_scale*sqrt(size(im,1)*size(im,2));
                end
                for c=1:size(cur_locs,1)
                    if parts.visible((cur_image_idx-1)*part_count + channel_ids(c))
                        x=cur_locs(c,1);
                        y=cur_locs(c,2);
                        if i>image_count
                            x=227-x;
                        end
                        ratio_x = 227.0 / size(im,2);
                        ratio_y = 227.0 / size(im,1);
                        x=int32(x/ratio_x);
                        y=int32(y/ratio_y);
                        x_min = max(x-box_size/2, 1);
                        x_max = min(x+box_size/2, size(im,2));
                        y_min = max(y-box_size/2, 1);
                        y_max = min(y+box_size/2, size(im,1));
                        batch_data = [batch_data; im(int32(y_min:y_max),int32(x_min:x_max),:)];
                        missing_data = [missing_data;false];
                    else
                        batch_data = [batch_data; [125]];
                        missing_data = [missing_data;true];
                    end
                end
            end
        end
        % Add spatial pyramid levels of image
        for l=pyramid_levels:-1:1
            x = fix(size(im,2)/(2^l));
            y = fix(size(im,1)/(2^l));
            if (x==0 || y==0)
                error('Image too small for spm');
            end
            xx=0;
            yy=0;
            while xx+x<=size(im,2)
                while yy +y <=size(im,1) 
                    batch_data = [batch_data;im(yy+1:yy+y,xx+1:xx+x,:)];
                    missing_data = [missing_data;false];
                    yy = yy+y;
                end        
                yy = 0;
                xx = xx+x;
            end
        end
        % Add the image
        batch_data = [batch_data; im];
        missing_data = [missing_data;false];
        tmp = caffe_features(batch_data,layer_image,mean_file,batch_size,crop_size)'; 
        features(i,:) = tmp(:);
    end
    
    if opts.store_features
        save([opts.cache_dir '/feats.mat'],'features','labels','tr_ID','-v7.3');    
    end
    
    ORR_total = ones(opts.repetitions,1);
    ARR_total = ones(opts.repetitions,1);
    for i=1:opts.repetitions
        if rand_tr_images>0
            [ tr_ID ] = createTrainTest( labels, rand_tr_images, rand_tr_part );
        end
        % Train and test
        model = train(labels(tr_ID,:),(features(tr_ID,:)),params);
        [pred,acc_cur,~] = predict(labels(~tr_ID,:),(features(~tr_ID,:)),model);

        % evaluate
        cm = confusionmat(labels(~tr_ID),pred);
        acc=sum(diag(cm))/sum(cm(:))*100;
        cm = cm./repmat(sum(cm,2),1,size(cm,2));
        map=nanmean(diag(cm)./sum(cm,2))*100;    
        ORR_total(i,1)=acc;
        ARR_total(i,1)=map;
        fprintf('Run %i ORR=%5.2f ARR=%5.2f\n',i, ORR_total(i,1),ARR_total(i,1));
    end
    fprintf('Mean over %i runs:\n',opts.repetitions);
    fprintf('ORR=%f +- %f\n',nanmean(ORR_total), nanstd(ORR_total));
    fprintf('ARR=%f +- %f\n',nanmean(ARR_total), nanstd(ARR_total));
end
