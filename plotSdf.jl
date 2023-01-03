using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

fit = y -> scale(
        interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
        range(0,1,length=length(y))
    )

width = [0.02, 0.07, 0.06, 0.048, 0.03, 0.019, 0.01]
thk = fit(width)
envelope = [0.2,0.21,0.23,0.4,0.88,1.0]
amp = fit(envelope)

function fish(thk, amp, k=5.3; L=2^6, A=0.1, St=0.3, Re=1e4)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdfFish(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))
    
    # parameters of the circle
    radius = L/10
    sdfCircle(x,t) = (norm2(x - [radius-2L+2, radius+L/2]) - radius) 

	# fish motion: travelling wave
	U = 1
	ω = 2π * St * U/(2A * L)

    function mapFish(x,t)
        xc = x
        return xc - SVector(0., A * L * amp(s(xc)) * sin(k*s(xc)-ω*t))
    end

    function mapCircle(x,t)
        return x - [t,0.]
    end

	function map(x, t)
		xc = x - [2L,L] # shift origin
		# if sdfFish(mapFish(xc,t),t) < 1
		# 	return mapFish(xc,t)
        # elseif sdfCircle(mapCircle(xc,t),t) < 1
        #     return mapCircle(xc,t)
        # else
        #     return xc + (mapFish(xc,t)-xc)*exp(-abs(sdfFish(mapFish(xc,t),t)/(sdfCircle(xc- [t,0.],t)))) + (mapCircle(xc,t)-xc)*exp(-abs(sdfCircle(xc- [t,0.],t)/(sdfFish(mapFish(xc,t),t))))
        # end
        if sdfFish(mapFish(xc,t),t) < sdfCircle(mapCircle(xc,t),t)
            if sdfFish(mapFish(xc,t),t) <= 100
                return mapFish(xc,t)
            else
                return mapFish(xc,t)/(sdfFish(mapFish(xc,t),t)-100)
            end
        else
            if sdfCircle(mapCircle(xc,t),t) <= 100
                return mapCircle(xc,t)
            else
                return mapCircle(xc,t)/(sdfCircle(mapCircle(xc,t),t)-100)
            end
        end
	end

    
    sdf(x,t) = minimum([sdfCircle(x,t), sdfFish(x,t)])

	# make the fish simulation
	return Simulation((6L+2,2L+2), [U,0.], L;
		Δt=0.025, ν=U*L/Re, body=AutoBody(sdf,map))
end

# Create the swimming shark
L,A,St = 3*2^5,0.1,0.3
swimmer = fish(thk, amp; L, A, St);

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 8*period, length=24*8)



function computeSDF(sim, t)
    s = copy(sim.flow.p)
    for I ∈ inside(s)
        x = loc(0, I)
        s[I] = sim.body.sdf(x,t*swimmer.L/swimmer.U)::Float64
    end
    contourf(s', clims=(-L,2L), linewidth=0,
            aspect_ratio=:equal, legend=true, border=:none)
end

@gif for t ∈ sim_time(swimmer) .+ cycle
    sim_step!(swimmer, t, remeasure=true, verbose=true)
    computeSDF(swimmer, t)
end

# grid = -1:0.05:2
# contourf(grid, grid, segment_sdf, clim=(-L,2L), linewidth=0)
# contour!(grid, grid, segment_sdf, levels=[0], color=:black)  # zero contour