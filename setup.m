function setup()    
    %% Get libs
    if ~exist('lib/SelectiveSearchCodeIJCV','file')
        getlib('http://koen.me/research/downloads/SelectiveSearchCodeIJCV.zip');
    else
        fprintf('Selective search exists already, skipping...\n');
    end
    na = dir('lib/liblinea*');
    if numel(na)==0
        getlib('http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip');
    else
        fprintf('liblinear exists already, skipping...\n');
    end
    if ~exist('lib/GetFullPath.m','file')
        getlib('http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28249/versions/8/download/zip')
        !rm lib/GetFullPath.c lib/InstallMex.m lib/Readme.txt lib/license.txt lib/uTest_GetFullPath.m
    else 
        fprintf('GetFullPath.m exists already, skipping...\n');
    end
    if ~exist('lib/vl_argparse.m','file')
        !wget --no-check-certificate -O lib/vl_argparse.m https://raw.githubusercontent.com/vlfeat/matconvnet/master/matlab/vl_argparse.m
    else
	fprintf('vl_argparse exists already, skipping...\n');
    end
    
    %% Get models
    required_files = {};
%     required_files = [required_files;{'cnn_finetuning/googlenet/model','http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel','405fc5acd08a3bb12de8ee5e23a96bec22f08204'}];
%     required_files = [required_files;{'cnn_finetuning/vgg19/model','http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel','239785e7862442717d831f682bb824055e51e9ba'}];
%     required_files = [required_files;{'cnn_finetuning/caffe_reference/model','http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel','4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46'}];

    for i=1:size(required_files,1)
        if exist(required_files{i,1},'file') 
            [~,sha1sum] = system(['sha1sum ' required_files{i,1} '  | awk ''{ print $1 }''']);
            if strcmp(strtrim(sha1sum), required_files{i,3})
                fprintf('%s exists already, skipping...\n',required_files{i,1});
                continue
            else
                fprintf('%s exists but is corrupt, downloading again...\n',required_files{i,1});
            end
        end
        if 0~=system(['wget -O ' required_files{i,1} ' ' required_files{i,2}]);
            error('Could not download file %s from %s\n',required_files{i,1},required_files{i,2});
        end
    end
    
    fprintf('\n\nSetup done, now clone caffe_pp and go to ./lib/ and compile all libraries and Matlab interfaces!\n');
    fprintf('1. ''git submodule update --init --recursive'' in the main folder\n');
    fprintf('2. ''make'' in ./lib/caffe_pp/\n');
    fprintf('3. ''make mat'' in ./lib/caffe_pp/\n');
    fprintf('4. ''make'' in ./lib/liblinear-2.1/\n');
    fprintf('5. ''make'' in ./lib/liblinear-2.1/matlab/\n');
end

function getlib(url)
    system('mkdir tmp');
    if 0~= system(['wget -O tmp/lib.zip ' url]) 
        error('Could download code');
    end
    if 0~= system('cd lib && unzip ../tmp/lib.zip ')
        error('Could not unzip');
    end
    if 0~= system('rm tmp/lib.zip')
        error('Could not remove temporary file');
    end
    if 0~= system('rmdir tmp')
        error('Could not remove directory tmp');
    end
end