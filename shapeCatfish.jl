using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

fit = y -> scale(
    interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
    range(0,1,length=length(y))
)

"""url2 = "C://Users//blagn771//Desktop//catfish.PNG"
catfish = load(url2)
plot(catfish)

nose, len = (120,470), 1050"""
width = [0.005, 0.05, 0.065, 0.07, 0.075, 0.078, 0.078, 0.075, 0.075, 0.0725, 0.072 ,0.0715,0.071,0.07,0.065,0.0625,0.0575,0.053,0.05,0.0475,0.0425,0.04,0.035,0.03,0.025,0.02,0.0175,0.0175,0.017,0.0165,0.016,0.0155,0.015,0.015,0.015,0.015,0.015,0.015]
"""scatter!(
    nose[1] .+ len .* range(0,1,length=length(width)),
    nose[2] .- len .* width, color= :blue, legend=false
)"""
thk = fit(width)
"""x = 0:0.01:1
plot!(
    nose[1] .+ len .* x,
    [nose[2] .- len .* thk.(x), nose[2] .+ len .* thk.(x)],
    color=:red
)"""

envelope = [0.051, 0.022, 0.0435, 0.0801, 0.133, 0.2225]./0.2225 # % of the tail amplitude for a catfish swimming at 1 BL/s (Anguilliform Locomotion across a Natural Range of Swimming Speeds)
amp = fit(envelope)

"""λ = 0.71
scatter(0:0.2:1, envelope)
colors = palette(:cyclic_wrwbw_40_90_c42_n256)
for t in 1/12:1/12:1
	plot!(x, amp.(x) .* sin.(2π/λ * x .- 2π*t),
		  color=colors[floor(Int,t*256)])
end
plot!(ylim=(-1.4,1.4), legend=false)"""

function fish(thk, amp, k=2*π/0.71; L=2^6, A=0.1, St=0.3, Re=5430)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdf(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))


	# fish motion: travelling wave
	ω = 2π * St * U/(2A * L)
	function map(x, t)
		xc = x - [L, L] # shift origin
		return xc - SVector(0., A * L * amp(s(xc)) * sin(k*s(xc)-ω*t))
	end

	# make the fish simulation
	return Simulation((3L+2,2L+2), [U,0.], L;
						ν=U*L/Re, body=AutoBody(sdf,map))
end

# Create the swimming catfish
L,A,St = 3*2^5,0.2225,0.63
U = 1
λ = 0.71
swimmer = fish(thk, amp, 2*π/λ; L, A, St);

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 23/24*period, length=24)

@gif for t ∈ cycle
	measure!(swimmer, t*swimmer.L/swimmer.U)
	contour(swimmer.flow.μ₀[:,:,1]',
			aspect_ratio=:equal, legend=false, border=:none)
end

# function plot_pressure(sim)
# 	contourf(sim.flow.p',
# 			 clims=(-1, 1), linewidth=0,
# 			 aspect_ratio=:equal, legend=true, border=:none)
# end

# # make a gif over a swimming cycle
# @gif for t ∈ sim_time(swimmer) .+ cycle
# 	sim_step!(swimmer, t, remeasure=true, verbose=false)
# 	# plot_vorticity(swimmer)
# 	plot_pressure(swimmer)
# end