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
        return x - [h,h]
    end

	# make the simulation
	Simulation((202,122), [1.,0.], radius1; ν=radius1/250, body=AutoBody(sdf,map))
end

swimmer = capsule(9.5,0.5,60)

function computeSDF(sim, t)
    s = copy(sim.flow.p)
    for I ∈ inside(s)
        x = loc(0, I)
        s[I] = sim.body.sdf(x,t*swimmer.L/swimmer.U)::Float64
    end
    contourf(s', clims=(-10,50), linewidth=0,
            aspect_ratio=:equal, legend=true, border=:none)
    savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end

@gif for t ∈ sim_time(swimmer)
    sim_step!(swimmer, t, remeasure=true, verbose=true)
    computeSDF(swimmer, t)
end