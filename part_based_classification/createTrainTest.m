function [ tr_ID ] = createTrainTest( labels, min_tr, min_tr_percentage )
    tr_ID=zeros(size(labels));
    for c=1:length(unique(labels))
        class_elements=find(labels==c);
        if ( length(class_elements) <= min_tr)
            tr_ID(class_elements)=1;
        else
            tr_elements=class_elements(randperm(length(class_elements),max(min_tr,ceil(min_tr_percentage*length(class_elements)))));
            tr_ID(tr_elements)=1;
        end
    end
    tr_ID=logical(tr_ID);
end

