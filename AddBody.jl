"""Deprecated"""

# function addBody(Bodies, L=100)
# 	"""addBody(Bodies::Array{SVector{Function, Function}}, L=100)
    
#         Bodies: array of SVector(sdf, map) for each of the independent bodies to add to the window.
#         L: carateristic dimension of the largest body, to create a great enough offset to delete the generetion of undesired body.

        
#     The default distance between two independent bodies is set to 100L. It impacts both their placement to not disturbe the other maps,
# 	in the function 'offset', and the selection of the second closest body to a given point in the function 'min_excluding_i'.
# 	The coefficients are computed to determine where each map should be used, therefore creating a global map that impacts the 
# 	whole simulation window.

# 	The output can directly be used as the body argument in the Simulation function provided by WaterLily."""

# 	sdfList, mapList = offset(Bodies, L)

# 	function min_excluding_i(sdfL, mapL, i, x, t)
# 		min_val = 100L
# 		for j in eachindex(sdfL)
# 			if j != i 
# 				val = sdfL[j](mapL[j](x,t),t)
# 				if val <= min_val
# 					min_val = val
# 				end
# 			end
# 		end
# 		return min_val
# 	end

# 	coef = [(x,t) -> μ₀(min_excluding_i(sdfList, mapList, i, x, t) - sdfList[i](mapList[i](x,t),t),1) for i in range(1, length(sdfList))]

# 	sdf(x,t) = minimum([sdfX(x,t) for sdfX in sdfList])
# 	map(x,t) = sum([mapList[i](x,t)*coef[i](x,t) for i in range(1, length(mapList))])

# 	return AutoBody(sdf, map)
# end