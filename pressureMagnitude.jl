using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2
using Statistics

fit = y -> scale(
        interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
        range(0,1,length=length(y))
    )

width = [0.005, 0.05, 0.065, 0.07, 0.075, 0.078, 0.078, 0.075, 0.075, 0.0725, 0.072 ,0.0715,0.071,0.07,0.065,0.0625,0.0575,0.053,0.05,0.0475,0.0425,0.04,0.035,0.03,0.025,0.02,0.0175,0.0175,0.017,0.0165,0.016,0.0155,0.015,0.015,0.015,0.015,0.015,0.015]
thk = fit(width)

envelope = [0.2,0.21,0.23,0.4,0.88,1.0]
amp = fit(envelope)

function fish(thk, amp, k=5.3; L=2^6, A=0.1, St=0.3, Re=5430)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdfFish(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))

	# fish motion: travelling wave
	U = 1
	ω = 2π * St * U/(2A * L)
	function map(x, t)
		xc = x - [2L,L] # shift origin
		if xc[2]< L/4
	
			return xc - SVector(0., A * L * amp(s(xc)) * sin(k*s(xc)-ω*t))
		else
			return xc - [t,0.]
		end
	end

    # parameters of the circle
    radius = L/10
    sdfCircle(x,t) = (norm2(x - [radius-2L+2, radius+L/2]) - radius) 

    sdf(x,t) = minimum([sdfCircle(x,t), sdfFish(x,t)])

	# make the fish simulation
	return Simulation((6L+2,2L+2), [U,0.], L;
		Δt=0.025, ν=U*L/Re, body=AutoBody(sdfFish,map))
end

# Create the swimming shark
L,A,St = 3*2^5,0.1,0.3
swimmer = fish(thk, amp; L, A, St);

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 23*period/24, length=24)

@gif for t ∈ cycle
	measure!(swimmer, t*swimmer.L/swimmer.U)
	contour(swimmer.flow.μ₀[:,:,1]',
			aspect_ratio=:equal, legend=true, border=:none)
end

# run the simulation a few cycles (this takes few seconds)
sim_step!(swimmer, 10, remeasure=true)
sim_time(swimmer)

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim)
	contourf(sim.flow.p',
			 clims=(-2, 2), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer)
end

@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	pressure = swimmer.flow.p'[2,:]
	scatter([i for i in range(1,length(pressure))], pressure,
		labels=permutedims(["pressure at L"]),
		xlabel="scaled distance",
		ylabel="scaled pressure",
		ylims=(-0.4,0.4))
end