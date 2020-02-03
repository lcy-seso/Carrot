function [outputArg1,outputArg2] = plot_one_cube(x_value,y_value,z_value, color)
    xslice = x_value;
    yslice = y_value;
    zslice = z_value;

    h_slice = slice(x,y,z,v,xslice,yslice,zslice);
    set(h_slice,'facealpha',0.05)
    set(h_slice,'facecolor',color)
end
