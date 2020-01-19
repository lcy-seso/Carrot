function [outputArg1,outputArg2] = plot_one_rhombus_cube(...
    xmin,xmax, ymin, ymax, zmin, zmax, color)

    % 8 vertex for the polygon.
    V = [zmin,xmin,xmin+ymin+zmin;...
         zmin,xmax,xmax+ymin+zmin;...
         zmin,xmax,xmax+ymax+zmin;...
         zmin,xmin,xmin+ymax+zmin;...
         zmax,xmin,xmin+ymin+zmax;...
         zmax,xmax,xmax+ymin+zmax;...
         zmax,xmax,xmax+ymax+zmax;...
         zmax,xmin,xmin+ymax+zmax];

    % 6 faces
    F = [1 2 3 4;5 6 7 8;...
         5 1 2 6;8 4 3 7;...
         5 1 4 8;6 2 3 7];
    h = patch('Faces',F,'Vertices',V);
    set(h,'facealpha',0.05)
    set(h,'facecolor',color)
end
