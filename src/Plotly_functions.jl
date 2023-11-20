using Plotly, Distributions, Copulas

function plotly_copula(copula, data::AbstractMatrix, function_type::AbstractString; col = "Viridis", title="", xlabel="u", ylabel="v", show_contours = true, scale = true)
    #Validate the type of function entered
    validate_function_type(function_type)
    
    colors_3D = ["Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric", "Greens", "Greys", "Hot", 
                    "Jet" , "Picnic", "Portland", "Rainbow", "RdBu", "Reds", "Viridis", "YlGnBu", "YlOrRd"]
    #Check copula length
    if length(copula) == 2
        if function_type == "scatter3d"
            error("Cannot use 'scatter3d' when copula length is 2.")
        end
    elseif length(copula) == 3
        if function_type != "scatter3d"
            error("You can only use 'scatter3d' when the copula length is 3.")
        end
    else
        error("The copula length must be 2 or 3.")
    end

    if function_type == "scatter"
        if col in colors_3D
            col = "lightblue"
        end
        return plotly_scatter(data, col, title, xlabel, ylabel)
    elseif function_type == "scatter+hist"
        if col in colors_3D
            col = "lightblue"
        end
        return plotly_scatter_hist(data, col)
    elseif function_type == "hist2D"
        if !(col in colors_3D)
            @warn("The color '$col' is not in Plotly's predefined list. 'YlGnBu' will be used as the default color.")
            println("Possible predefined colors in Plotly:")
            println(colors_3D)
            col = "YlGnBu"
        end
        return plot_histogram_2D(data, col, title, xlabel, ylabel)
    elseif (function_type == "pdf_contours" || function_type == "cdf_contours")
        if !(col in colors_3D)
            @warn("The color '$col' is not in Plotly's predefined list. 'Viridis' will be used as the default color.")
            println("Possible predefined colors in Plotly:")
            println(colors_3D)
            col = "Viridis"
        end
        return plotly_contours(copula, function_type, col, title, xlabel, ylabel, scale)  
    elseif function_type == "scatter3d"
        return plotly_scatter3D(copula, data, col)
    else
        #Check if the color is in the predefined list
        if !(col in colors_3D)
            @warn("The color '$col' is not in Plotly's predefined list. 'Viridis' will be used as the default color.")
            println("Possible predefined colors in Plotly:")
            println(colors_3D)
            col = "Viridis"
        end
        return plotly_surface(copula, function_type, col, title, xlabel, ylabel, show_contours, scale)
    end
end

function validate_function_type(function_type)
    if !(function_type in ["pdf", "cdf", "scatter", "scatter+hist", "hist2D", "pdf_contours", "cdf_contours", "scatter3d"])
        error("The function must be 'pdf', 'cdf', 'scatter', 'scatter+hist', 'hist2D', 'pdf_contours', 'cdf_contours', 'scatter3d'")
    end
end

function plotly_scatter(data, col, title, xlabel, ylabel)
    scatter_plotly = Plotly.scatter(
        x=data[1, :],
        y=data[2, :],
        mode="markers",
        marker=attr(color=col, size=3)
    )

    layout = create_layout(title, xlabel, ylabel)
    
    return Plotly.plot(scatter_plotly, layout)
end

function plotly_scatter_hist(data, col)
    trace1 = Plotly.scatter(x=data[1, :], y=data[2, :], mode="markers", marker=attr(color=col, size=3))
    trace2 = Plotly.histogram(x=data[1, :], xaxis="x", yaxis="y2", marker=attr(color="#c7f6d4"))
    trace3 = Plotly.histogram(y=data[2, :], xaxis="x2", yaxis="y", marker=attr(color="#caacf9"))

    layout = create_scatter_hist_layout()
    
    return Plotly.plot([trace1, trace2, trace3], layout)
end

function plotly_histogram_2D(data, col, title, xlabel, ylabel)
    layout = create_hist_2D_layout(title, xlabel, ylabel)
    return Plotly.plot(histogram2d(x=data[1, :], y=data[2, :], colorscale=col),layout)
end

function plotly_contours(copula, function_type, col, title, xlabel, ylabel, scale)
    u = range(0, stop=1, length=100)
    v = range(0, stop=1, length=100)
    grid = Iterators.product(u, v)

    z_data = [calculate_value(copula, ui, vi, function_type) for (ui, vi) in grid]
    z_data = reshape(z_data, 100, 100)
    layout = create_countors_layout(title, xlabel, ylabel)
    return Plotly.plot(contour(x=u,y=v, z=z_data, colorscale=col, showscale=scale),layout)
end

function plotly_surface(copula, function_type, col, title, xlabel, ylabel, show_contours, scale)
    u = range(0, stop=1, length=100)
    v = range(0, stop=1, length=100)
    grid = Iterators.product(u, v)

    z_data = [calculate_value(copula, ui, vi, function_type) for (ui, vi) in grid]
    z_data = reshape(z_data, 100, 100)

    layout = create_surface_layout(title, xlabel, ylabel, show_contours, scale, col, function_type)

    return Plotly.plot(surface(x=u, y=v, z=z_data, contours_z=show_contours, colorscale=col, showscale=scale), layout)
end

function calculate_value(copula, ui, vi, function_type)
    if (function_type == "pdf" || function_type == "pdf_contours")
        return pdf(copula, [ui, vi])
    elseif (function_type == "cdf" || function_type == "cdf_contours")
        return cdf(copula, [ui, vi])
    end
end

function plotly_scatter3D(copula, data, col)
    layout = Layout(
        scene = attr(
        xaxis_title = "u",
        yaxis_title = "v",
        zaxis_title = "w"
    ))
    return Plotly.plot(scatter3d(x = data[1, :], y = data[2, :], z = data[3, :],
    mode = "markers", 
    marker = attr(size = 3, color = data[3, :], colorscale=col
    )),layout)
end

function create_layout(title, xlabel, ylabel)
    return Plotly.Layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=create_axis_properties(),
        yaxis=create_axis_properties()
    )
end

function create_scatter_hist_layout()
    return Plotly.Layout(
        autosize=false,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=create_axis_properties(zeroline=false, domain=[0, 0.81]),
        yaxis=create_axis_properties(zeroline=false, domain=[0, 0.81]),
        xaxis2=create_axis_properties(zeroline=false, domain=[0.85, 1]),
        yaxis2=create_axis_properties(zeroline=true, domain=[0.85, 1]),
        height=600,
        width=600,
        bargap=0,
        showlegend=false
    )
end

function create_hist_2D_layout(title, xlabel, ylabel)
    return Plotly.Layout(
        title = title,
        xaxis=attr(title=xlabel, showgrid=false, zeroline=false),
        yaxis=attr(title=ylabel, showgrid=false, zeroline=false),
        autosize=false,
        height=500,
        width=500,
        hovermode="closest",
        showlegend=false
    )
end

function create_surface_layout(title, xlabel, ylabel, show_contours, scale, col, function_type)
    contours_settings = show_contours ? attr(show=true, usecolormap=true, project_z=true, showlines=true) : false

    zaxis_title = function_type == "pdf" ? "PDF" : (function_type == "cdf" ? "CDF" : "function_type")

    return Plotly.Layout(
        title=title,
        autosize=false,
        scene_camera_eye=attr(x=1.87, y=0.88, z=-0.64),
        width=500,
        height=500,
        margin=attr(l=65, r=50, b=65, t=90),
        scene=attr(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zaxis_title),
        xaxis=create_axis_properties(),
        yaxis=create_axis_properties(),
        showlegend=false
    )
end

function create_axis_properties(; zeroline=true, domain=[0, 1], showgrid=true, gridcolor="#FAFAFA", gridwidth=0.5, griddash="solid", showline=true, linecolor="black")
    return attr(
        zeroline=zeroline,
        domain=domain,
        showgrid=showgrid,
        gridcolor=gridcolor,
        gridwidth=gridwidth,
        griddash=griddash,
        showline=showline,
        linecolor=linecolor
    )
end

function create_countors_layout(title, xlabel, ylabel)
  return Plotly.Layout(
    title = title,
    xaxis = attr(title = xlabel, showgrid = false, zeroline = false),
    yaxis = attr(title = ylabel, showgrid = false, zeroline = false),
    autosize = false,
    height = 500,
    width = 500,
    hovermode = "closest",
    showlegend = false
    )
end


