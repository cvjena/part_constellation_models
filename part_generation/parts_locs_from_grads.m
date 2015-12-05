function [ part_locs ] = parts_locs_from_grads(opts)
    imagedir= opts.imagedir;
    %read image list
    fid=fopen(opts.imagelist_file,'r');
    imagelist=textscan(fid,'%s');
    imagelist=imagelist{1};
    fclose(fid);
    % layer
    layer = opts.part_layer;
    part_count = opts.part_layer_channel_count;
    
    mean_file = opts.mean_mat_file;
    batch_size = opts.batch_size;
    crop_size = opts.crop_size;
    deploy = opts.deploy;
    model = opts.model;
        
    parfor (i=1:opts.parfor_workers, opts.parfor_arg)
        matcaffe_init(1,deploy,model,1,mod(i,opts.gpu_count));
    end
    
    fprintf('%s\n',datestr(now));
    % The estimated part locations for all images and parts
    part_locs=nan(size(imagelist,1), part_count,2);
    parfor (i = 1:size(imagelist,1), opts.parfor_arg)
        if opts.verbose_output
            fprintf('Image %i: %s\n',i, imagelist{i});
        end
        g=caffe_gradients(imread([imagedir '/' imagelist{i}]),layer,(1:part_count)',mean_file,batch_size,crop_size);
        for p=1:part_count
            %read gradient map
%             gmap=load(sprintf('%s%s/gradient_layer%s_channel%i.mat',basedir, imagelist{i},layer, p-1));
%             gmap=gmap.gradient_map;
            gmap = squeeze(sum(abs(g(:,:,:,p)),3));
            if sum(isnan(gmap(:))) >0 || sum(gmap(:)~=0)<1
                continue
            end
            [est_x,est_y]=fitGMMToGradient(zeros(crop_size,crop_size,3),gmap,[],2);
%             imshow(gmap,[])
%             hold all
%             plot(est_x,est_y,'X','MarkerSize',20,'LineWidth',10)
%             ginput(1)
            part_locs(i,p,:)=[est_x,est_y];
        end
    end
    part_locs = convert_locs_to_CUB200_format(part_locs);
    save(opts.part_loc_file,'part_locs');
end