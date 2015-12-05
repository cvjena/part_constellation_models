function [ x,y ] = fitGMMToGradient(imagepath, gmap,bbox, num_clusters )
%fitGMMToGradient:
%   bbox = [col row width height]


%     d=load(gradient_path);
%     gmap=d.gradient_map;
    img=zeros(227,227,3);%imread(imagepath);
% %     set gradient outside of bounding box to 0
%     bbox_orig=bbox;
%     rect_size=min(size(img(:,:,1)));
%     ratio = max(227.0 / size(img,1), 227.0 / size(img,2));
%     bbox=int32(floor(bbox*ratio));
%     h_offset = ceil(size(img,1)*ratio - 227) / 2;
%     w_offset = ceil(size(img,2)*ratio - 227) / 2;
%     bbox(1)=max(w_offset+1,bbox(1));
%     bbox(2)=max(h_offset+1,bbox(2));
%     bbox(3)=bbox(3)-max(w_offset-bbox(1),0);
%     bbox(4)=bbox(4)-max(h_offset-bbox(2),0);
%     bbox(1)=bbox(1)-w_offset+1;
%     bbox(2)=bbox(2)-h_offset+1;
%     bbox(3)=min(227-bbox(1),bbox(3));
%     bbox(4)=min(227-bbox(2),bbox(4));
%     mask = ones(size(gmap));
%     mask(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))=0;
%     gmap(logical(mask))=0;

    if (false)
        % simplify calculation
        gmap(gmap<quantile(gmap(:),0.9))=0;
        [rows,cols,vals]=find(gmap);
        if (size(vals,1)<2)
            x=NaN;
            y=NaN;
            return;
        end
        [~,model,~]=weightedemgm([rows cols]',num_clusters,[vals],3,100);
        % Reorder accoring to weight
        [~,argidx]=sort(-model.weight);
        model.mu=model.mu(:,argidx);
        model.Sigma=model.Sigma(:,:,argidx);
        model.weight=model.weight(argidx);

        est_x=model.mu(2,1);
        est_y=model.mu(1,1);
    elseif (false)
        [est_y,est_x]=find(max(gmap(:))==gmap);
        est_y=est_y(1);
        est_x=est_x(1);
    elseif (true)
        if any(gmap(:)~=0)
            % Create the gaussian filter with hsize = [5 5] and sigma = 2
            G = fspecial('gaussian',[20 20],3);
            % Filter it
            gmap = imfilter(gmap,G,'same');
            [est_y,est_x]=find(max(gmap(:))==gmap,1,'last');
            est_y=est_y(1);
            est_x=est_x(1);
        else
            est_y=-1;
            est_x=-1;
        end
    end
    
    
    
    x=est_x;
    y=est_y;
    % calc ratio 
    ratio_x = 227.0 / size(img,2);
    ratio_y = 227.0 / size(img,1);
    % we add 0.5 to the converted result to avoid numerical problems.
%     h_offset = (size(img,1)*ratio - 227) / 2;
%     w_offset = (size(img,2)*ratio - 227) / 2;
%     x=est_x+w_offset;
%     y=est_y+h_offset;
    x=int32(x/ratio_x);
    y=int32(y/ratio_y);
    
%     % Display result
% %     show assignment to cluster
%     figure;
% %     map=double(full(sparse(rows,cols,labels',size(gmap,1),size(gmap,2))));
%     gmap=gmap/max(gmap(:));
%     imshow(gmap);
%     hold on;
%     plot(est_x,est_y,'x','MarkerSize',20,'LineWidth',3);
%     hold off;
%     ginput(1);
%     close all;


%     figure;
%     gmap=gmap/max(gmap(:));
%     imshow(gmap);%<quantile(gmap(:),0.9)
%     colors=['b','r','g'];
%     for i=1:size(model.mu,2)
%         h=plot_gaussian_ellipsoid(flipud(model.mu(:,i)),rot90(model.Sigma(:,:,i),2));
%         set(h,'color',colors(i));
%     end
%     ginput(1);
%     close
    
    
%     figure;
%     img_cropped=imread(sprintf('/home/simon/tmp/cub200-maps/%s/inputimage.jpg',imagepath));
%     imshow(img_cropped);
%     hold all;
%     plot(est_x,est_y,'x','MarkerSize',20,'LineWidth',3);
%     figure;
%     imshow(img);
%     hold all;
%     plot(x,y,'x','MarkerSize',20,'LineWidth',3);
end


function [bbox]=transformBbox(img,bbox)
    % calc ratio 
    ratio = max(227.0 / size(img,1), 227.0 / size(img,2));
    % we add 0.5 to the converted result to avoid numerical problems.
    h_offset = (size(img,1)*ratio - 227) / 2;
    w_offset = (size(img,2)*ratio - 227) / 2;
    bbox(1)=bbox(1)+w_offset;
    bbox(2)=bbox(2)+h_offset;
    bbox=int32(bbox/ratio);
end
