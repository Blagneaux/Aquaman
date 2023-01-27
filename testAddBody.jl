using WaterLily

@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))

@fastmath function Gurvan(Bodies)
	# Create a different offset for each of the bodies so that when the general map is computed, there is no problem of
	# body spontaneous generation. This offset is then substracted when the individual map is applied where it needs to be thus
	# hiding its existence.
	
    sdfList = [ (x,t) -> offsetSdf(x,t,i) for i in 1:length(Bodies)]
    mapList = [ (x,t) -> offsetMap(x,t,i) for i in 1:length(Bodies)]

	function offsetSdf(x,t, i)
		xc = x + [0.,(-1)^i * 100*i * 100]
		return Bodies[i][1](xc,t)
	end

	function offsetMap(x,t, i)
		xc = x - [0.,(-1)^i * 100*i * 100]
		return Bodies[i][2](xc,t)
	end

	return (sdfList, mapList)
end

function addBody(Bodies)
	# The default distance between two independent bodies is set to 10000. It impacts both their placement to not disturbe the other maps,
	# in the function 'Gurvan', and the selection of the second closest body to a given point in the function 'min_excluding_i'

	sdfList, mapList = Gurvan(Bodies)

	function min_excluding_i(sdfL, mapL, i, x, t)
		min_val = 10000
		for j in eachindex(sdfL)
			if j != i 
				val = sdfL[j](mapL[j](x,t),t)
				if val <= min_val
					min_val = val
				end
			end
		end
		return min_val
	end

	coef = [(x,t) -> μ₀(min_excluding_i(sdfList, mapList, i, x, t) - sdfList[i](mapList[i](x,t),t),1) for i in range(1, length(sdfList))]

	sdf(x,t) = minimum([sdfX(x,t) for sdfX in sdfList])
	map(x,t) = sum([mapList[i](x,t)*coef[i](x,t) for i in range(1, length(mapList))])

	return AutoBody(sdf, map)
end