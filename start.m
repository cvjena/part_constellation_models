function start(varargin)
%%% Configuration
%% Dependencies
opts.caffe_path='./lib/caffe_pp';
opts.liblinear_path='./lib/liblinear-2.1/matlab/';
opts.selsearch_path='./lib/SelectiveSearchCodeIJCV/';

%% Datasets
% If the imagepaths are relative to some imagedir, specify it here, otherwise leave empty
opts.imagedir='/';
% Path to your dataset, where the imagelist, the labels and the train test split is
opts.basedir = '/path/to/your/dataset/';

% Cache files
opts.cache_dir = [pwd '/cache/'];
opts.store_features = false;

%% CNN model
% Here you can choose your CNN model, have a look in the folder cnn_finetuning for possible models
opts.cnn_dir = [fileparts(mfilename('fullpath')) '/cnn_finetuning/caffe_reference/'];
% The batch size when calculating features, should match the value in the deploy.prototxt file
opts.batch_size = 11;
% The net image input size as specified in the deploy.prototxt file
opts.crop_size = 227;

%% Fine-tuning of CNN
opts.finetuning = true;
opts.finetuning_iters = 10000;
opts.finetuning_gpu = 0;
% Proposal generation 
% We generate object proposals and filter those for fine-tuning
% You can specify, if the bounding box should be estimated as well
% This might take additional time when generating the proposals
opts.estimate_bbox = true;
% Calculate the image mean of all proposals
opts.calculate_mean = true;

%% Part model hyperparameters
opts.part_layer = 'pool5';
opts.part_layer_channel_count = 256;
% The number of parts per view
opts.no_selected_parts = 10;
% Number of visible parts
opts.no_visible_parts = 5;
% Number of views per class
opts.view_count = 10;
% We initialize the part model randomly and hence we repeat optimization several
% times and take the best optimum
opts.iterations = 5;
% Select whether to compute a part model for each class separately
opts.class_wise = true;

%% Classification parameters
% The CNN layer to take the features from
opts.feature_global = 'relu7';
% Use flipped images in training as well
opts.use_flipped = true;
% Add features from the estimated bounding box
opts.use_bounding_box = true;
% Calculate features on a spatial pyramid of the input image, define the depth here
opts.pyramid_levels = 0;
% If you want to use random splits into training and test, specify this here
% Absolute number of training images per class
opts.rand_tr_images = -1;
% Relative part of each class to be used for training
opts.rand_tr_part = 0;
% How many splits should be evaluated
opts.repetitions = 1;

% Set this if you want to use parts in training
opts.use_parts = true;
% At each estimated part location, we extract a patch of size sqrt(part_scale*width*height) to calculate featureres
% Multiple scales can be defined like [0.44 0.24]
opts.part_scales = [0.44 0.24];
% The CNN layer to take the features for the parts from
opts.feature_part = 'relu7';
% Parameters for SVM training, you should use estimate these parameters using cross-validation on training data
opts.svm_params='-q';% VGG19: -s 2 -c 0.0000432', GoogLeNet: -c 0.0009766  -s 2 -q'

%% Parallelism
% Number of parallel workers
opts.parfor_workers = 2;
% Specify if you want to use parallelism at all
opts.use_parfor = true;

% Set to false if you want to hide almost all unnecessary output
opts.verbose_output = true;

%% Depended paths
function opts = setDependentPaths(opts)
    % List of images 
    opts.imagelist_file = [opts.basedir '/imagelist_absolute.txt'];
    % The assignment to train and test, mandatory if finetuning is used! 
    % 1 - train, 0 - test
    opts.tr_ID_file = [opts.basedir '/tr_ID.txt'];
    % List of class labels starting from 1
    opts.labels_file = [opts.basedir '/labels.txt'];

    % CNN stuff
    opts.deploy = [opts.cnn_dir '/deploy.prototxt'];
    opts.model = [opts.cnn_dir '/model'];
    opts.mean_mat_file = [opts.cnn_dir '/mean.mat'];
    opts.mean_proto_file = [opts.cnn_dir '/mean.binaryproto'];

    opts.caffe_executable = [opts.caffe_path '/build/tools/caffe'];

    % Finetuning output dir: Derive from last two folders of opts.cnn_dir
    dd = strsplit(opts.cnn_dir,'/');
    dd = dd(~cellfun(@isempty,dd));
    opts.finetuning_dir = [opts.cache_dir '/' strjoin(dd(end-1:end),'/')];
end

%% Parse arguments
addpath('lib');
opts = setDependentPaths(opts);
opts = vl_argparse(opts,varargin);
% Again: Parse arguments to allow setting of the dependent paths
opts = setDependentPaths(opts);
opts = vl_argparse(opts,varargin);


%% Make all paths for finetuning absolute
opts.model = GetFullPath(opts.model);
opts.caffe_executable = GetFullPath(opts.caffe_executable);
opts.cnn_dir = GetFullPath(opts.cnn_dir);

%% Post processing of the params
% Preparation
addpath(genpath('./'));
addpath([opts.caffe_path '/matlab/caffe']);
addpath(opts.liblinear_path);
addpath(opts.selsearch_path);
% Cache files
mkdir(opts.cache_dir);
work_dir=pwd;cd(opts.cache_dir);opts.cache_dir=pwd;cd(work_dir);
opts.part_loc_file = [opts.cache_dir '/part_locs.mat'];
opts.est_bbox_file = [opts.cache_dir '/est_bbox.txt'];
opts.caffe_window_file_train = [opts.cache_dir '/windows_train.txt'];
opts.caffe_window_file_val = [opts.cache_dir '/windows_val.txt'];
opts.caffe_part_model = [opts.cache_dir '/part_model.mat'];

opts

%% The algorithm
if opts.use_parfor
  opts.parfor_arg = opts.parfor_workers;
else
  opts.parfor_arg = 0;
end
opts.gpu_count = gpuDeviceCount;

% Extract parts
% Result will be stored in cache file
% IMPORTANT: part locs are always relative to the normalized image of width
% and height 227
if exist(opts.part_loc_file)
    fprintf('Loading part locs from %s\n',opts.part_loc_file);
    load(opts.part_loc_file,'part_locs');
else
    fprintf('Calculating part locations for all channels and all images...\n');
    fprintf('IMPORTANT: This will take quite some time, especially with large nets like VGG19\n' );
    part_locs = parts_locs_from_grads(opts);
end

% Learn part model
% This function outputs the whole part model, but we only use the first
% output which are the ids of the opts.no_selected_parts most often used parts
if exist(opts.caffe_part_model)
    fprintf('Loading part model from cache...\n');
    load(opts.caffe_part_model,'channel_ids','part_visibility');
else
    fprintf('Calculating part model...\n');
    [ channel_ids, part_visibility] = evaluate_part_locs_anchor_multiview(part_locs, load(opts.tr_ID_file), ...
            load(opts.labels_file), opts.no_selected_parts, opts.no_visible_parts, opts.view_count, opts.iterations);
    save(opts.caffe_part_model,'channel_ids','part_visibility');
end
    
% Part based localization to generate region proposals for fine-tuning as
% well as bounding boxes for each image
if (~opts.estimate_bbox || exist(opts.est_bbox_file)) && exist(opts.caffe_window_file_train) && ...
        exist(opts.caffe_window_file_val) 
    fprintf('Loading estimated bounding boxes and region proposals from disk...\n');
elseif ~opts.use_bounding_box && ~opts.finetuning
    fprintf('Region proposals for fine-tuning or estimated bboxes are not needed, skipping...\n');
else
    fprintf('Generating region proposals and estimated bounding boxes for CNN finetuning...\n');
    selsearch_object_detector( channel_ids(1:opts.no_selected_parts), part_locs, part_visibility, opts );
end

% Fine-tuning
if opts.finetuning
    opts.model = [opts.finetuning_dir '/model_ft_iter_' int2str(opts.finetuning_iters) '.caffemodel'];
    opts.deploy = [opts.finetuning_dir '/deploy_ft.prototxt'];
    if exist(opts.model) && exist(opts.deploy)
        fprintf('Using pretrained model...\n');
    else
        fprintf('\nTime for fine-tuning of the CNN! This might take some time...\n');
%         fprintf('\nHit enter to continue\n');
%         pause
        % Make sure labels start at 1 
        labels=load(opts.labels_file);
        labels=labels-min(labels(:))+1;
        finetuning(max(labels(:)),opts);
    end
end

% Classification
fprintf('Starting classification...\n');
part_box_classification_multiscale( channel_ids(1:opts.no_selected_parts), part_locs, opts );
end