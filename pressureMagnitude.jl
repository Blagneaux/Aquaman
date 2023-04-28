using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

using CSV, Tables

# include("capsule.jl")
include("cambridgeFish.jl")

fit = y -> scale(
    interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
    range(0,1,length=length(y))
)

width = [0.02, 0.06, 0.06, 0.05, 0.03, 0.015, 0.01]
thk = fit(width)

_nthread = Threads.nthreads()
if _nthread==1
    @warn "WaterLily.jl is running on a single thread.\n
Launch Julia with multiple threads to enable multithreaded capabilities:\n
    \$julia -t auto $PROGRAM_FILE"
else
    print("WaterLily.jl is running on ", _nthread, " thread(s)\n")
end

# Variables to plot the different gif implemented
momentOrdreZero = false
pressureMapFull = true
pressureOnWalls = false
pressionAtOnePoint = false

# Physic variables of the problem
L = 71.2 - 6.5 # length of the fish, taking into account the head design
A = 0.4663076581549986 # relative amplitude of the motion of the tail
St = 0.611392 # Strouhal number, corresponds to the frequency of the motion of the tail
U = 1.574 # velocity scale
v = 1.574 #0.915 velocity
n = 640*2+2 #642 #3*2^10+2 # length of the taking
m = 258 # width of the tank
Re = 12875 #6070 # Reynolds number

f = St * U/(2A * L)

function wall(a, b)
    function sdf(x,t)
        xa = x-a
        ba = b-a
        h = clamp(dot(xa,ba)/dot(ba,ba), 0.0, 1.0)
        return norm2(xa - ba*h) - 40
    end

    function map(x,t)
        xc = x - [0, m/2]
        return xc
    end

    return AutoBody(sdf, map)
end

# capsuleShape = capsule(L, St, A, v; n, m)
fishShape = fish(thk; L=81.8*4, Re=6070, n, m)

wallShape1 = wall([-n,140], [n,140])
wallShape2 = wall([-n,-140], [n,-140])

# swimmerBody = capsuleShape + wallShape1 + wallShape2
swimmerBodyCambridge = fishShape + wallShape1 + wallShape2

swimmer = Simulation((n,m), [0.,0.], (81.8*4), U=U; ν=U*(81.8*4)/Re, body=swimmerBodyCambridge)

# Save a time span for one swimming cycle
period = 2A/St
nb_snapshot = 24*8
cycle = range(0, 0.5*period*23/24, length=nb_snapshot)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

if momentOrdreZero
	@gif for t ∈ cycle
		measure!(swimmer, t*swimmer.L/swimmer.U)
		contour(swimmer.flow.μ₀[:,:,1]',
				aspect_ratio=:equal, legend=true, border=:none)
		plot!(Shape([511,582,582,511],[129,129,130,130]), legend=false, c=:red, opacity=1)
		plot!(Shape([527,527,527,527],[123,123,135,135]), legend=false, c=:red, opacity=1)
		plot!(Shape([0,642,642,0],[0,0,29,29]), legend=false, c=:black, opacity=0.2)
		plot!(Shape([0,642,642,0],[229,229,258,258]), legend=false, c=:black, opacity=0.2)
		savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
	end
end

pressureFull = []
pressurInf = []

# plot the pressure scaled by the body length L and flow speed U
function plot_pressure(sim, t)
	fish = ones(size(sim.flow.p))
	for I ∈ inside(sim.flow.p)
		if sim.body.sdf(WaterLily.loc(0,I),t*sim.L/sim.U) < 0
			fish[I] = 0
		end
	end

	modeP∞ = "classique"

	pressureₜ = sim.flow.p

	if modeP∞ == "classique"
		P∞ = pressureₜ[2,Int(m/2)]
		pressureₜ .-= P∞

	elseif modeP∞ == "smart"
		if t*sim.L/sim.U <= 2/f
			P∞ = (pressureₜ[2,30]+pressureₜ[2,m-29])/2
		else
			P∞ = (pressureₜ[n-1,30]+pressureₜ[n-1,m-29])/2
		end
		pressureₜ .-= P∞

	elseif modeP∞ == "norm"
		filterPressure = vec(pressureₜ[2:end-1,2:end-1].*fish[2:end-1,2:end-1])
		Pmax = sort(union(filterPressure))[end]
		if Pmax == 0 && length(sort(union(filterPressure))) > 1
			Pmax = sort(union(filterPressure))[end-1]
		end
		Pmin = 0
		P∞ = 0
		if Pmax != 0
			Pmin = sort(union(filterPressure))[1]
			if Pmin == 0
				Pmin = sort(union(filterPressure))[2]
			end
			pressureₜ .-= Pmin
			pressureₜ ./= (Pmax - Pmin)
		end
		print("max pressure ", Pmax, '\n')
		print("min pressure ", Pmin, '\n')
	end


	# The indentation depends on the length and duration of the sim
	append!(pressureFull,vec(pressureₜ.*fish))
	
	# pressureFull[:,trunc(Int,1+ceil(0.02*ceil(t*sim.L/sim.U/5)+t*sim.L/sim.U/5))] .= vec(pressureₜ.*fish)
	append!(pressurInf,[P∞])

	contourf(color=palette([:blue,:lightgrey,:red],9),
			(pressureₜ[2:n-1,2:m-1]'.*fish[2:n-1,2:m-1]'), 
			linewidth=0, connectgaps=false, dpi=300, aspect_ratio=1,
			clims=(-1, 1), legend=true, border=:none)

	# The origin of the map is on the bottom left
	plot!(Shape([0,n,n,0],[0,0,29,29]), legend=false, c=:black, opacity=0.2)
	plot!(Shape([0,n,n,0],[m-29,m-29,m,m]), legend=false, c=:black, opacity=0.2)

	plot!(Shape([3,3,3,3],[127,127,131,131]), legend=false, c=:black)
	plot!(Shape([1,5,5,1],[129,129,129,129]), legend=false, c=:black)

	plot!(Shape([n-290,n-290,n-290,n-290],[m-57,m-57,m-61,m-61]), legend=false, c=:black)
	plot!(Shape([n-292,n-288,n-288,n-292],[m-59,m-59,m-59,m-59]), legend=false, c=:black)

	# plot!(Shape([1,n-600,n-600,1],[1,1,m,m]), legend=false, c=:black, opacity=0.5)
	plot!(Shape([n-556,n-556,n-556,n-556],[1,1,m,m]), legend=false, c=:black)
	plot!(Shape([n-51,n-51,n-51,n-51],[1,1,m,m]), legend=false, c=:black)
	plot!(Shape([n-157*4,n-157*4,n-157*4,n-157*4],[1,1,m,m]), legend=false, c=:black)

	# print(t,"\n")

  	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t*U)*".png")
end


if pressureMapFull
	# make a gif over a swimming cycle
	@gif for t ∈ sim_time(swimmer) .+ cycle
		sim_step!(swimmer, t, remeasure=true, verbose=true)
		plot_pressure(swimmer, t)
	end

	CSV.write("C:/Users/blagn771/Desktop/FullPressure.csv", Tables.table(reshape(pressureFull, :, nb_snapshot)), writeheader=false)
	CSV.write("C:/Users/blagn771/Desktop/FullPressureInf.csv", Tables.table(pressurInf), writeheader=false)
end

pressureTop = zeros(n,24*8)
pressureBottom = zeros(n,24*8)

if pressureOnWalls
	@gif for t ∈ sim_time(swimmer) .+ cycle
		sim_step!(swimmer, t, remeasure=true, verbose=true)
		pressure1 = swimmer.flow.p'[29,:]
		pressure2 = swimmer.flow.p'[198,:]

		pressureₜ = swimmer.flow.p
		P∞ = pressureₜ[3,Int(m/2)]

		pressureTop[:,trunc(Int,round(t*St*24/2/A)+1)] .= pressure1.-P∞
		pressureBottom[:,trunc(Int,round(t*St*24/2/A)+1)] .= pressure2.-P∞

		scatter([i for i in range(1,length(pressure1))], [pressure1.-P∞, pressure2.-P∞],
			labels=permutedims(["pressure on the bottom wall", "pressure on the top wall sensor"]),
			xlabel="scaled distance",
			ylabel="scaled pressure",
			ylims=(-2,2))
		savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
	end

	# CSV.write("C:/Users/blagn771/Desktop/TopPressure.csv", Tables.table(pressureTop), writeheader=false)
	# CSV.write("C:/Users/blagn771/Desktop/BottomPressure.csv", Tables.table(pressureBottom), writeheader=false)
end

f = St * U/(2A * L)

function mapAngle(t)
	if t <= 4/f
		amp = 25*π/180
		α = amp*sin(2π*f*t)
		print(α*180/π)
		return α
	end
	α = 0
	return α
end

pressureExtraction = []
angleExtraction = []

if pressionAtOnePoint
	@gif for t ∈ sim_time(swimmer) .+ cycle
		angle = mapAngle(t*swimmer.L/swimmer.U)
		sim_step!(swimmer, t, remeasure=true, verbose=true)

		pressurePoint = swimmer.flow.p'[m-30-29,n-290]

		pressureₜ = swimmer.flow.p
		nb_iter = swimmer.pois.n

		P∞ = pressureₜ[3,Int(m/2)]

		append!(pressureExtraction,[pressurePoint - P∞])
		append!(angleExtraction,[angle])

		plot([i for i ∈ range(1,length(pressureExtraction))],[pressureExtraction,angleExtraction],
		labels=permutedims(["pressure at the extraction point (290mm, 30mm)","angle of the tail"]),
		xlabel="scaled time",
		xlims=(0,192),
		ylims=(-1.5,0.5),
		ylabel="scaled pressure")
		plot!(twinx(),nb_iter, xlims=(1,9142))
	end
end