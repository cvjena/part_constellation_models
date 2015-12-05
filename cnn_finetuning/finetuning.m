function [  ] = finetuning( num_classes, opts )
    olddir = pwd;
    mkdir(opts.finetuning_dir);
    % Adjust and copy proto files 
    if 0~=system(['sed ''s/##NUM_CLASSES##/' int2str(num_classes+1) '/g'' ''' opts.cnn_dir '/train_val_template.prototxt'' > ''' opts.finetuning_dir '/train_val.prototxt'''])
        error('Error creating train_val.prototxt')
    end
    if 0~=system(['sed ''s/##NUM_CLASSES##/' int2str(num_classes+1) '/g'' ''' opts.cnn_dir '/deploy_template.prototxt'' > ''' opts.finetuning_dir '/deploy_ft.prototxt'''])
        error('Error creating train_val.prototxt')
    end
    if 0~=system(['sed ''s/##MAX_ITER##/' int2str(opts.finetuning_iters) '/g'' ''' opts.cnn_dir '/solver_template.prototxt'' > ''' opts.finetuning_dir '/solver.prototxt'''])
        error('Error creating solver.prototxt')
    end
    if 0~=system(['cp ''' opts.mean_proto_file ''' ''' opts.finetuning_dir '/mean.binaryproto'''])
        error('Error creating solver.prototxt')
    end
    if 0~=system(['sed ''s/##MAX_ITER##/' int2str(opts.finetuning_iters) '/g'' ''' opts.cnn_dir '/solver_template.prototxt'' > ''' opts.finetuning_dir '/solver.prototxt'''])
        error('Error creating solver.prototxt')
    end
    
    fprintf(['\n\nNow open a bash, go to ' opts.finetuning_dir ' and run:\n']);
    fprintf(['# ' opts.caffe_executable ' train -solver=solver.prototxt -weights=''' opts.cnn_dir '/model'' -gpu=' int2str(opts.finetuning_gpu) ' \n']);
    fprintf('Hit enter when training has finished!');
    input('','s');
    cd(opts.finetuning_dir);
%     if opts.verbose_output
%         outputfile = '';
%     else
%         outputfile = ' 2> /dev/null';
%     end
%     if 0~=system([opts.caffe_executable ' train -solver=solver.prototxt -weights=''' opts.cnn_dir '/model'' -gpu=' int2str(opts.finetuning_gpu) ' ' outputfile])
%         cd(olddir)
%         error('Caffe training failed.')
%     end
    if 0~=system(['rm ./*.solverstate'])
        warning('Did not delete any solverstate files.')
    end
    cd(olddir)
end

