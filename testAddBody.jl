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

function createFishAlone(thk, amp, k=5.3; L=2^6, A=0.1, St=0.3, Re=1e4)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdf(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))

	# fish motion: travelling wave
	U = 1
	ω = 2π * St * U/(2A * L)
	function map(x, t)
		xc = x - [2L,L] # shift origin
		return xc - SVector(0., A * L * amp(s(xc)) * sin(k*s(xc)-ω*t))
	end

	# make the fish body
	return AutoBody(sdf,map)
end


function createCircleAlone(radius)
	# fish geometry: thickened line SDF
	sdf(x,t) = (norm2(x - [radius-2L+2, radius+L/3]) - radius) 

	# fish motion: travelling wave
	function map(x, t)
		xc = x - [2L,L] # shift origin
		return xc - [t,0.]
	end

	# make the circle body
	return AutoBody(sdf,map)
end


function fish(thk, amp, k=5.3; L=2^6, A=0.1, St=0.3, Re=5430)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	function sdfFish(x,t)
        # xc = x - [0., 100L]
		xc = x
        return √sum(abs2, xc - L * SVector(s(xc), 0.)) - L * thk(s(xc))
    end

	# fish motion: travelling wave
	U = 1
	ω = 2π * St * U/(2A * L)
    
    # parameters of the circle
    radius = L/5
    sdfCircle(x,t) = (norm2(x - [radius+2-2L, radius+L/2 - 100L]) - radius) 

    function mapFish(x,t)
		xc = x
        # xc = x + [0., 100L]
		# xc = xc + [t,0.]
        return xc - SVector(0., A * L * amp(s(xc)) * sin(k*s(xc)-ω*t))
    end

    function mapCircle(x,t)
        return x - [0., 100L]
    end
	
	@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
    μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))

    function map(x, t)
		xc = x - [2L,L] # shift origin
        len∿ = 1

        coef🐠 = μ₀(sdfCircle(mapCircle(xc,t),t)-sdfFish(mapFish(xc,t),t),len∿)
        coef🔴 = μ₀(sdfFish(mapFish(xc,t),t)-sdfCircle(mapCircle(xc,t),t),len∿)

		return mapCircle(xc,t)*coef🔴 + mapFish(xc,t)*coef🐠
    end

    sdf(x,t) = minimum([sdfCircle(x,t), sdfFish(x,t)])

	# make the fish simulation
	return Simulation((6L+2,2L+2), [U,0.], L; U,
			Δt=0.025, ν=U*L/Re, body=AutoBody(sdfFish,mapFish))
end

@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))

function addBody(mainSimu, Bodies, dims::Tuple, u_BC::Vector; 
					Δt=0.25, ν=0., uλ::Function=(i,x)->u_BC[i])
	sdfList = [[(x,t) -> mainSimu.body.sdf.sdf(x + [0., 100*mainSimu.L],t)] [(x,t) -> body.sdf.sdf(x+ [0.,100n*(-1)ⁿmainSimu.L],t) for (n, body) in enumerate(Bodies)]]
	mapList = [[(x,t) -> mainSimu.body.map(x - [0., 100mainSimu.L],t)] [(x,t) -> body.map(x - [0.,100n*(-1)ⁿmainSimu.L],t) for (n, body) in enumerate(Bodies)]]
	coef = [μ₀(minimum([deleteat!(sdfList[i],j)(deleteat!(mapList[i],j)(x,t),t) for i in range(length(sdfList)-1)]) - sdfList[j](mapList[j](x,t),t),mainSimu.ϵ) for j in range(length(sdfList))]

	sdf = minimum(sdfList)
	map = mapList*coef'

	return Simulation(dims, u_BC, mainSimu.L; 
			mainSimu.U, Δt, ν, mainSimu.ϵ, uλ, body=AutoBody(sdf, map))
end

# Create the swimming shark
L,A,St = 3*2^5,0.1,0.3
swimmerFish = fish(thk, amp; L, A, St);

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 8*period, length=24*8)

swimmer = addBody(swimmerFish, [createCircleAlone(20)], (6L+2,2L+2), [1, 0.], ν=1*L/5430)

# @gif for t ∈ cycle
# 	measure!(swimmer, t*swimmer.L/swimmer.U)
# 	contour(swimmer.flow.μ₀[:,:,1]',
# 			aspect_ratio=:equal, legend=true, border=:none)
# end

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

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