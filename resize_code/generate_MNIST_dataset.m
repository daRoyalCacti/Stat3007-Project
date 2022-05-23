save_images('test/digitStruct.mat', './test/', '../dataset/test_images_MNIST/', '../dataset/test_labels_MNIST.txt', '../dataset/test_no_digits.txt');
save_images('train/digitStruct.mat', './train/', '../dataset/training_images_MNIST/', '../dataset/training_labels_MNIST.txt', '../dataset/training_no_digits.txt');
save_images('extra/digitStruct.mat', './extra/', '../dataset/extra_images_MNIST/', '../dataset/extra_labels_MNIST.txt', '../dataset/extra_no_digits.txt');


function save_images(input_data, input_image_dir, image_dir, class_file, no_digits_file)
    load(input_data);

    fileID = fopen(class_file, 'w');
    fileID2 = fopen(no_digits_file, 'w');
    counter = 0;
%     for ii = 1:3
        
    for ii = 1:length(digitStruct)
        data = digitStruct(ii);
        
        %writing the number of digits in this image
        fprintf(fileID2, '%i\n', length(data.bbox));
        
        
        
        for jj = 1:length(data.bbox)
            counter = counter + 1;
            %writing the digits 
            img = imread([input_image_dir, data.name]);

            n = data.bbox(jj).label;
            if n == 10
                n = 0;
            end
            num = num2str(n);
            
            fprintf(fileID, [num, '\n']);

            %getting the corners of the bounding boxes
            ulx = data.bbox(jj).left;
            uly = data.bbox(jj).top;
            lrx = data.bbox(jj).left+data.bbox(jj).width;
            lry = data.bbox(jj).top+data.bbox(jj).height;


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

            imw = 15;
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
               for ll = 1:size(img_cropped2,1)
                    for kk = 1:size(img_cropped2,2)
                        %using nearest neighbour
                        old_x = round(ll/size(img_cropped2,1)*size(img_cropped,1));
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


                        img_cropped2(ll,kk,:) = img_cropped(old_x, old_y,:);
                    end
                end



                img_cropped = img_cropped2;
            end



            if size(img_cropped,1) ~= imh || size(img_cropped,2) ~= imw
                fprintf('Image %i is the wrong size\n', counter)
            end

            ii_Padded = sprintf( '%05d', counter ) ;
            imwrite( img_cropped, [image_dir, ii_Padded, '.png'])
        end
    end
    fclose(fileID);
    fclose(fileID2);
end