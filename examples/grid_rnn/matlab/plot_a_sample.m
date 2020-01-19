function plot_a_sample(M,N,P,plane_color,point_color)
    zmin = 0;
    zmax = M; % z axis

    xmin = 1;
    xmax = N; % x axis

    ymin = 1;
    ymax = P; % y axis

    %%
    grid on
    box on
    hold on
    axis on
    view(3)

    xlim([1, N + 1])
    ylim([1, P + 1])
    zlim([0, M])
    
    xlabel('source sequence')
    ylabel('target sequence')
    zlabel('depth')

    %%
    peaks_x = [xmin, xmax, xmax, xmin];
    peaks_y = [ymin, ymin, ymax, ymax];
    peaks_z = [zmin, zmin, zmin, zmin];
    patch(peaks_x, peaks_y, peaks_z, plane_color);
    alpha(0.3);

    peaks_x = [xmin, xmax, xmax, xmin];
    peaks_y = [ymin, ymin, ymax, ymax];
    peaks_z = [zmax-1, zmax-1, zmax-1, zmax-1];
    patch(peaks_x, peaks_y, peaks_z, plane_color);
    alpha(0.1);

    %% 
    peaks_x = [xmin, xmin, xmin, xmin];
    peaks_y = [ymin, ymax, ymax, ymin];
    peaks_z = [zmin, zmin, zmax-1, zmax-1];
    patch(peaks_x, peaks_y, peaks_z, plane_color);
    alpha(0.3);

    peaks_x = [xmax, xmax, xmax, xmax];
    peaks_y = [ymin, ymax, ymax, ymin];
    peaks_z = [zmin, zmin, zmax-1, zmax-1];
    patch(peaks_x, peaks_y, peaks_z, plane_color);
    alpha(0.3);

    %%
    peaks_x = [xmin, xmax, xmax, xmin];
    peaks_y = [ymax, ymax, ymax, ymax];
    peaks_z = [zmin, zmin, zmax-1, zmax-1];
    patch(peaks_x, peaks_y, peaks_z, plane_color);
    alpha(0.3);

    peaks_x = [xmin, xmax, xmax, xmin];
    peaks_y = [ymin, ymin, ymin, ymin];
    peaks_z = [zmin, zmin, zmax-1, zmax-1];
    patch(peaks_x, peaks_y, peaks_z, plane_color);
    alpha(0.3);

    %% perpendicular to z axis
    [x, y] = meshgrid(1:N, 1:P);

    c = 1;
    point_x = zeros(P*N,1);
    point_y = zeros(P*N,1);
    point_z = zeros(P*N,1);

    for k = 0:M-1
        z = k*ones(P, N);
        for i = 1:P
            for j = 1:N
                point_x(c,1) = x(i,j);
                point_y(c,1) = y(i,j);
                point_z(c,1) = z(i,j);
                c = c+1;
            end
        end
        scatter3(point_x, point_y, point_z,...
            'Marker','o','LineWidth', 5,...
            'MarkerEdgeColor', point_color,...
            'MarkerFaceColor', point_color)

        surf(x,y,z)
    end
end