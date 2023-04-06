using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

using CSV, Tables

include("capsule.jl")

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

# Physic variables of the problem
L = 71.2 - 6.5 # length of the fish, taking into account the head design
A = 0.4663076581549986 # relative amplitude of the motion of the tail
St = 0.611392 # Strouhal number, corresponds to the frequency of the motion of the tail
U = 1 # Speed of the fish
n = 642 # length of the taking
m = 258 # width of the tank

function wall(a, b)
    function sdf(x,t)
        xa = x-a
        ba = b-a
        h = clamp(dot(xa,ba)/dot(ba,ba), 0.0, 1.0)
        return norm2(xa - ba*h) - 40
    end

    function map(x,t)
        xc = x - [5L, m/2]
        return xc
    end

    return AutoBody(sdf, map)
end

capsuleShape = capsule(L, St, A, U; n, m)

wallShape1 = wall([-600,140], [400,140])
wallShape2 = wall([-600,-140], [400,-140])

swimmerBody = capsuleShape + wallShape1 + wallShape2

swimmer = Simulation((n,m), [0.,0.], (L+6.5), U=U; ν=U*(L+6.5)/6070, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 8*period*23/24, length=24*8)

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

pressureFull = zeros(n*m,24*8)
pressurInf = []

# plot the pressure scaled by the body length L and flow speed U
function plot_pressure(sim, t)
	fish = ones(size(sim.flow.p))
	for I ∈ inside(sim.flow.p)
		if sim.body.sdf(WaterLily.loc(0,I),t*sim.L/sim.U) < 0
			fish[I] = 0
		end
	end

	pressureₜ = sim.flow.p
	P∞ = pressureₜ[3,Int(m/2)]
	pressureₜ .-= P∞

	# The indentation depends on the length and duration of the sim
	pressureFull[:,trunc(Int,1+ceil(0.02*ceil(t*sim.L/sim.U/5)+t*sim.L/sim.U/5))] .= vec(pressureₜ.*fish)
	append!(pressurInf,[P∞])

	contourf(color=palette([:blue,:lightgrey,:red],9),
			(pressureₜ[2:641,2:257]'.*fish[2:641,2:257]'), 
			linewidth=0, connectgaps=false, dpi=300,
			clims=(-2, 2), legend=true, border=:none)
	plot!(Shape([0,642,642,0],[0,0,29,29]), legend=false, c=:black, opacity=0.2)
	plot!(Shape([0,642,642,0],[229,229,258,258]), legend=false, c=:black, opacity=0.2)

	plot!(Shape([3,3,3,3],[127,127,131,131]), legend=false, c=:black)
	plot!(Shape([1,5,5,1],[129,129,129,129]), legend=false, c=:black)

	plot!(Shape([n-365,n-365,n-365,n-365],[57,57,61,61]), legend=false, c=:black)
	plot!(Shape([n-367,n-363,n-363,n-367],[59,59,59,59]), legend=false, c=:black)

	plot!(Shape([1,n-600,n-600,1],[1,1,m,m]), legend=false, c=:black)
	plot!(Shape([n-556,n-556,n-556,n-556],[1,1,m,m]), legend=false, c=:black)
	plot!(Shape([n-51,n-51,n-51,n-51],[1,1,m,m]), legend=false, c=:black)

	print(t,"\n")

  	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


if pressureMapFull
	# make a gif over a swimming cycle
	@gif for t ∈ sim_time(swimmer) .+ cycle
		sim_step!(swimmer, t, remeasure=true, verbose=false)
		plot_pressure(swimmer, t)
	end

	CSV.write("C:/Users/blagn771/Desktop/FullPressure.csv", Tables.table(pressureFull), writeheader=false)
	CSV.write("C:/Users/blagn771/Desktop/FullPressureInf.csv", Tables.table(pressurInf), writeheader=false)
end

pressureTop = zeros(n,24*8)
pressureBottom = zeros(n,24*8)

if pressureOnWalls
	@gif for t ∈ sim_time(swimmer) .+ cycle
		sim_step!(swimmer, t, remeasure=true, verbose=true)
		pressure1 = swimmer.flow.p'[30,:]
		pressure2 = swimmer.flow.p'[228,:]

		sum₁ = sum(pressure1)
		sum₂ = sum(pressure2)
		len₁₂ = length(pressure1)-2

		pressureTop[:,trunc(Int,round(t*St*24/2/A)+1)] .= pressure1.-(sum₁/len₁₂)
		pressureBottom[:,trunc(Int,round(t*St*24/2/A)+1)] .= pressure2.-(sum₂/len₁₂)

		scatter([i for i in range(1,length(pressure1))], [pressure1.-(sum₁/len₁₂), pressure2.-(sum₂/len₁₂)],
			labels=permutedims(["pressure on the bottom wall", "pressure on the top wall"]),
			xlabel="scaled distance",
			ylabel="scaled pressure",
			ylims=(-2,2))
		savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
	end

	CSV.write("C:/Users/blagn771/Desktop/TopPressure.csv", Tables.table(pressureTop), writeheader=false)
	CSV.write("C:/Users/blagn771/Desktop/BottomPressure.csv", Tables.table(pressureBottom), writeheader=false)
end