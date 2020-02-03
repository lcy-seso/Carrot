function plot_hyperplane(xmin,xmax, ymin, ymax, zvalue, color)
   % 4 vertex for the hyperplan.
    V = [xmin,ymin,zvalue;...
         xmax,ymin,zvalue;...
         xmax,ymax,zvalue;...
         xmin,ymax,zvalue];
    % 1 faces
    F = [1 2 3 4];
    h = patch('Faces',F,'Vertices',V);
    set(h,'facealpha',0.05)
    set(h,'facecolor',color)
end
