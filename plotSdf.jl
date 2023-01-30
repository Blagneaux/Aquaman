using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

include("AddBody.jl")

fit = y -> scale(
        interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
        range(0,1,length=length(y))
    )

width = [0.02, 0.07, 0.06, 0.048, 0.03, 0.019, 0.01]
thk = fit(width)
envelope = [0.2,0.21,0.23,0.4,0.88,1.0]
amp = fit(envelope)

function circle(radius, offset)
	# circle geometry
	sdf(x,t) = (norm2(x - [radius+offset[1], radius+offset[2]]) - radius) 

	# circle motion
	function map(x, t)
		xc = x - [2L,L] # shift origin
		return xc - [t,0.]
	end

	# make the circle body
	return SVector(sdf,map)
end

function fish(thk, amp, k=5.3; L=2^6, A=0.1, St=0.3, Re=1e4)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdf(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))

	# fish motion: travelling wave
	U = 1
	ω = 2π * St * U/(2A * L)
	function map(x, t)
		xc = x - [2L,L]# shift origin
		return xc - SVector(0., A * L * amp(s(xc)) * sin(k*s(xc)-ω*t))
	end

	# make the fish body
	return SVector(sdf,map)
end

# Create the swimming shark
L,A,St = 3*2^5,0.1,0.3
fishBody = fish(thk, amp; L, A, St)
circleBody = circle(20, [-2L+2,L/2])
circleBody2 = circle(10,[-2L+2,-3L/4])

swimmerBody = addBody([fishBody, circleBody, circleBody2])
swimmer = Simulation((6L+2,2L+2), [U,0.], L; U, Δt=0.025, ν=U*L/5430, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 23*period/3, length=24*8)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

# function computeSDF(sim, t)
#     s = copy(sim.flow.p)
#     for I ∈ inside(s)
#         x = loc(0, I)
#         s[I] = sim.body.sdf(x,t*swimmer.L/swimmer.U)::Float64
#     end
#     contourf(s', clims=(-L/2,2L), linewidth=0,
#             aspect_ratio=:equal, legend=true, border=:none)
#     savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
# end

# @gif for t ∈ sim_time(swimmer) .+ cycle
#     sim_step!(swimmer, t, remeasure=true, verbose=true)
#     computeSDF(swimmer, t)
# end

@gif for t ∈ cycle
	measure!(swimmer, t*swimmer.L/swimmer.U)
	contour(swimmer.flow.μ₀[:,:,1]',
			aspect_ratio=:equal, legend=true, border=:none)
end

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.σ',
			 color=palette(:roma), clims=(-1, 1), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
    savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer,t)
end