function [ part_locs2 ] = convert_locs_to_CUB200_format( part_locs )

%     load('part_locs_caffe.mat','part_locs');
    [image_count, part_count, ~] = size(part_locs);
    % Convert to CUB200 format
    part_locs2=nan(image_count* part_count,5);
    for i = 1:image_count
%         fprintf('Image %i\n',i);
        for p=1:part_count
            if any(isnan(part_locs(i,p,:)))
                part_locs2((i-1)*part_count+p,:)=[i p -1 -1 0];
            else
                part_locs2((i-1)*part_count+p,:)=[i p reshape(part_locs(i,p,:),1,2) 1];
            end
        end
    end
%     part_locs = part_locs2;
%     save('part_locs_caffe.mat','part_locs');
end

