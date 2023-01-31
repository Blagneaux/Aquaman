using WaterLily
using LinearAlgebra: norm2

function capsule(radius1, radius2, h)
	# geometry
	function sdf(x,t)
        xc = [x[1], abs(x[2])]
        b = (radius1 - radius2)/h
        a = sqrt(1 - b^2)
        k = dot(xc,[a,-b])
        if k < 0
            return norm2(xc) - radius1
        elseif k > a*h
            return norm2(xc - [h, 0]) - radius2
        else
            return dot(xc, [b, a]) - radius1
        end
    end
    
    function map(x,t)
        xc = x - [7h,258/2]
        return xc + [t, 0]
    end

	# make the simulation
	return SVector(sdf,map)
end