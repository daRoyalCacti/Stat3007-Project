save_images('test/digitStruct.mat', './test/', '../dataset/test_images/', '../dataset/test_labels.txt');
save_images('train/digitStruct.mat', './train/', '../dataset/training_images/', '../dataset/training_labels.txt');
save_images('extra/digitStruct.mat', './extra/', '../dataset/extra_images/', '../dataset/extra_labels.txt');

% input_data = 'test/digitStruct.mat';
% image_dir = '../dataset/test_images/';
% class_file = '../dataset/test_labels.txt';
function save_images(input_data, input_image_dir, image_dir, class_file)
    load(input_data);

    fileID = fopen(class_file, 'w');
    for ii = 1:length(digitStruct)
        data = digitStruct(ii);

        %getting the ordering of the digits
        %want to sort them by lower left x coordinate
        left = [data.bbox.left];
        [~, I] = sort(left, 'ascend');

        %writing the digits in the order found above
        img = imread([input_image_dir, data.name]);
        num = '';
        for jj = 1:length(data.bbox)
            n = data.bbox(I(jj)).label;
            if n == 10
                n = 0;
            end
            num = [num, num2str(n)];
        end
        fprintf(fileID, [num, '\n']);

        %getting the corners of the bounding boxes
        ulx = inf;
        uly = inf;
        lrx = -inf;
        lry = -inf;
        for jj = 1:length(data.bbox)
            ulx = min(ulx, data.bbox(jj).left);
            uly = min(uly, data.bbox(jj).top);
            lrx = max(lrx, data.bbox(jj).left+data.bbox(jj).width);
            lry = max(lry, data.bbox(jj).top+data.bbox(jj).height);
        end
        
        if ulx < 1 
            ulx = 1;
        end
        if ulx > size(img, 2)
            ulx = size(img, 2);
        end
        if lrx < 1 
            lrx = 1;
        end
        if lrx > size(img, 2)
            lrx = size(img, 2);
        end
        if uly < 1
            uly = 1;
        end
        if uly > size(img, 1)
            uly = size(img, 1);
        end
        if lry < 1
            lry = 1;
        end
        if lry > size(img, 1)
            lry = size(img, 1);
        end
        
        imw = 60;
        imh = 30;
       
        
        %resizing the images to be imwximh
        if (lrx-ulx) <= imw && lry-uly <=imh && size(img,2) >= imw && size(img,1) >= imh
           %if the bounding box is too small
           % - just extend the bounding box
           extend_x = imw - (lrx-ulx)-1; %the amount x needs to be extended
           extend_y = imh - (lry-uly)-1; %the amount y needs to be extended

           exl = floor(extend_x/2); %extend left
           exr = ceil(extend_x/2);  %extend right
           exu = floor(extend_y/2); %extend up
           exd = ceil(extend_y/2);  %extend down

           if ulx - exl > 0 && lrx + exr <= size(img, 2)     %if there is room to extend x
               ulx = ulx - exl;
               lrx = lrx+exr;
           else
               if ulx - exl > 0 %if there is room to the right
                    lrx = size(img, 2);
                    ulx = size(img, 2) - imw+1;
               else %if there is room to extend to the left
                    ulx = 1;
                    lrx = imw;
               end
           end


           if uly - exu > 0 && lry + exd <= size(img, 1)     %if there is room to extend y
               uly = uly - exu;
               lry = lry+exd;
           else
               if uly - exu > 0 %if there is room to the up
                    lry = size(img, 1);
                    uly = size(img, 1) - imh+1;
               else %if there is room to extend to the down
                    uly = 1;
                    lry = imh;
               end
           end


            %cropping the image
            img_cropped = img(uly:lry, ulx:lrx, :);
        else
            
            %make the bounding box the right aspect ratio

            w = lrx-ulx;
            h = lry-uly;

            ds = imw/imh;   %the desired aspect ratio

            if ds*h > w    %if the image is taller than longer
                %extend the image in the x-direction
                imh2 = h;
                imw2 = floor(h*ds);
                
                if imw2 > size(img, 2)
                    imw2 = w;
                    imh2 = floor(w/ds);
                    if imh2 > size(img, 1)  %if the resizing is still to big
                        %need to decrease the size of the bounding box
                        %just making the image thinner
                        imh2 = size(img,1);
                        imw2 = floor(imh2*ds);
                    end
                end
            else    %if the image is longer than it is tall
                %extend the image in the y-direction
                imw2 = w;
                imh2 = floor(w/ds);
                if imh2 > size(img, 1)
                    imh2 = h;
                    imw2 = floor(h*ds);
                    if imw2 > size(img, 2)  %if the resizing is still to big
                        %need to decrease the size of the bounding box
                        %just making the image shorter
                        imw2 = size(img,2);
                        imh2 = floor(imw2/ds);
                    end
                end
            end

            %extending the bounding box
            extend_x = imw2 - (lrx-ulx)-1; %the amount x needs to be extended
            extend_y = imh2 - (lry-uly)-1; %the amount y needs to be extended

            exl = floor(extend_x/2); %extend left
            exr = ceil(extend_x/2);  %extend right
            exu = floor(extend_y/2); %extend up
            exd = ceil(extend_y/2);  %extend down

            if ulx - exl > 0 && lrx + exr <= size(img, 2)     %if there is room to extend x
                ulx = ulx - exl;
                lrx = lrx+exr;
            else
                if ulx - exl > 0 %if there is room to the right
                    lrx = size(img, 2);
                    ulx = size(img, 2) - imw2+1;
               else %if there is room to extend to the left
                    ulx = 1;
                    lrx = imw2;
               end
           end


           if uly - exu > 0 && lry + exd <= size(img, 1)     %if there is room to extend y
               uly = uly - exu;
               lry = lry+exd;
           else
               if uly - exu > 0 %if there is room to the up
                    lry = size(img, 1);
                    uly = size(img, 1) - imh2+1;
               else %if there is room to extend to the down
                    uly = 1;
                    lry = imh2;
               end
           end

           %cropping the image
           img_cropped = img(uly:lry, ulx:lrx, :);

           %downsampling the cropped image
           img_cropped2 = uint8(zeros( imh, imw, 3 ));
           for jj = 1:size(img_cropped2,1)
                for kk = 1:size(img_cropped2,2)
                    %using nearest neighbour
                    old_x = round(jj/size(img_cropped2,1)*size(img_cropped,1));
                    old_y = round(kk/size(img_cropped2,2)*size(img_cropped,2));

                    if old_x == 0
                        old_x = 1;
                    end
                    if old_x >= size(img_cropped,1)
                        old_x = size(img_cropped,1);
                    end
                    if old_y == 0
                        old_y = 1;
                    end
                    if old_y >= size(img_cropped,2)
                        old_y = size(img_cropped,2);
                    end


                    img_cropped2(jj,kk,:) = img_cropped(old_x, old_y,:);
                end
            end



            img_cropped = img_cropped2;
        end



        if size(img_cropped,1) ~= imh || size(img_cropped,2) ~= imw
            fprintf('Image %i is the wrong size\n', ii)
        end

        ii_Padded = sprintf( '%05d', ii ) ;
        imwrite( img_cropped, [image_dir, ii_Padded, '.png'])

    end
    fclose(fileID);
end