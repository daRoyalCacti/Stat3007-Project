input_dir = './MNIST/';
output_dir = '../dataset/MNIST/';
label_file = '../dataset/MNIST_labels.txt';

dirs = {'Test/', 'Train/'};


fileID = fopen(label_file, 'w');
counter = 1;
for ii = 1:length(dirs)
    for jj = 0:9
        curr_dir = [input_dir, dirs{ii}, num2str(jj), '/'];
        all_images = dir([curr_dir, '*.png']);
        for kk = 1:length(all_images)
            img = imread( [all_images(kk).folder, '/', all_images(kk).name] );
            ii_Padded = sprintf( '%05d', counter ) ;
            
            %cropping the images to be as small as possible
            for ll = 1:28
                if mean(img(:, ll)) ~= 255
                    l = ll;
                    break
                end
            end
            
            for ll = 28:-1:1
                if mean(img(:, ll)) ~= 255
                    r = ll;
                    break
                end
            end
            
            for ll = 1:28
                if mean(img(ll, :)) ~= 255
                    t = ll;
                    break
                end
            end
            
            for ll = 28:-1:1
                if mean(img(ll, :)) ~= 255
                    b = ll;
                    break
                end
            end
            
            small_img = img(t:b, l:r);
            
                
            %expanding the images to make the small image the right aspect
            %ratio
            new_img = ones(2*(r-l+1), r-l+1 );
            bw = abs(floor( (2*(r-l+1) - (b-t) )/2));
            if bw == 0
                bw = 1;
            end
            
            cc = 1;
            for ll = bw:bw+size(small_img,1)-1
                new_img(ll, :) = small_img(cc, :)/255;
                cc = cc + 1;
            end
            

           %downsampling the new image
           img_scaled = zeros( 30, 15 );
           for ll = 1:size(img_scaled,1)
                for mm = 1:size(img_scaled,2)
                    %using nearest neighbour
                    old_x = round(ll/size(img_scaled,1)*size(new_img,1));
                    old_y = round(mm/size(img_scaled,2)*size(new_img,2));

                    if old_x == 0
                        old_x = 1;
                    end
                    if old_x >= size(new_img,1)
                        old_x = size(new_img,1);
                    end
                    if old_y == 0
                        old_y = 1;
                    end
                    if old_y >= size(new_img,2)
                        old_y = size(new_img,2);
                    end


                    img_scaled(ll,mm,:) = new_img(old_x, old_y,:);
                end
            end
            
            imwrite( img_scaled, [output_dir, ii_Padded, '.png'])
            
            fprintf(fileID, [num2str(jj), '\n']);
            
            counter = counter + 1;
        end
    end
end
fclose(fileID);

