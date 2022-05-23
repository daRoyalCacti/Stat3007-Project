% decolorize code from http://www.eyemaginary.com/Rendering/TurnColorsGrayAlgorithm.pdf 
% based on the paper from https://www.sciencedirect.com/science/article/abs/pii/S0031320306004766?via%3Dihub

inputs = {'../dataset/test_images/', '../dataset/training_images/', '../dataset/extra_images/'};
outputs = {'../dataset/test_images_grayscale/', '../dataset/training_images_grayscale/', '../dataset/extra_images_grayscale/'};

for kk = 1:length(inputs)
    input = inputs{kk};
    output = outputs{kk};

    a=dir([input, '*.png']);
    for ii = 1:size(a,1)
        ii_str = sprintf( '%05d', ii );
        img = im2double(imread([input, ii_str, '.png']));

        tones= decolorize(img,0.1,3,0.01);

        imwrite( tones, [output, ii_str, '.png'])
    end

end