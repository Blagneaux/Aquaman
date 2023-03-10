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

function wall(a, b)
    function sdf(x,t)
        xa = x-a
        ba = b-a
        h = clamp(dot(xa,ba)/dot(ba,ba), 0.0, 1.0)
        return norm2(xa - ba*h) - 40
    end

    function map(x,t)
        xc = x - [5L, 258/2]
        return xc
    end

    return SVector(sdf, map)
end

L,A,St,U = 71.2-6.5,0.466,0.61,0.89
capsuleShape = capsule(L, St, A)

wallShape1 = wall([-600,140], [400,140])
wallShape2 = wall([-600,-140], [400,-140])

swimmerBody = addBody([capsuleShape, wallShape1, wallShape2])
# swimmerBody = AutoBody(capsuleShape[1], capsuleShape[2])

swimmer = Simulation((642,258), [0.,0.], (L+6.5), U=0.89; ν=U*(L+6.5)/6070, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, period*8*23/24, length=24*8)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

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

pressureFull = zeros(642*258,24*8)
moyFull = zeros(24*8)

# plot the pressure scaled by the body length L and flow speed U
function plot_pressure(sim, t)
	fish = ones(size(sim.flow.p))
	for I ∈ inside(sim.flow.p)
		if sim.body.sdf(WaterLily.loc(0,I),t*sim.L/sim.U) < 0
			fish[I] = 0
		end
	end

	pressureₜ = sim.flow.p.*fish


	sumₚ = sum(pressureₜ)
    len_inside = max(length(filter(x -> x!=0, pressureₜ)),1)
    print("sum ",sumₚ,"\n")
    print("len ", len_inside, "\n")
    moyₚ = sumₚ/len_inside
    print("moy ", moyₚ, "\n")

    pressureₜ = pressureₜ.-moyₚ

	pressureFull[:,trunc(Int,round(t*St*24/2/A)+1)] .= vec(pressureₜ.*fish)
	moyFull[trunc(Int,round(t*St*24/2/A)+1)] = moyₚ

	contourf(pressureₜ'.*fish', connectgaps=false,
			 clims=(-2, 2), legend=true, border=:none)
	plot!(Shape([0,642,642,0],[0,0,29,29]), legend=false, c=:black)
	plot!(Shape([0,642,642,0],[229,229,258,258]), legend=false, c=:black)
  	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


# # make a gif over a swimming cycle
# @gif for t ∈ sim_time(swimmer) .+ cycle
# 	sim_step!(swimmer, t, remeasure=true, verbose=false)
# 	plot_pressure(swimmer, t)
# end

CSV.write("C:/Users/blagn771/Desktop/FullPressure.csv", Tables.table(pressureFull), writeheader=false)
CSV.write("C:/Users/blagn771/Desktop/FullPressureMoy.csv", Tables.table(moyFull), writeheader=false)

# pressureTop = zeros(642,24*8)
# pressureBottom = zeros(642,24*8)

# @gif for t ∈ sim_time(swimmer) .+ cycle
# 	sim_step!(swimmer, t, remeasure=true, verbose=true)
# 	pressure1 = swimmer.flow.p'[30,:]
# 	pressure2 = swimmer.flow.p'[228,:]

# 	sum₁ = sum(pressure1)
# 	sum₂ = sum(pressure2)
# 	len₁₂ = length(pressure1)-2

# 	pressureTop[:,trunc(Int,round(t*St*24/2/A)+1)] .= pressure1.-(sum₁/len₁₂)
# 	pressureBottom[:,trunc(Int,round(t*St*24/2/A)+1)] .= pressure2.-(sum₂/len₁₂)

# 	scatter([i for i in range(1,length(pressure1))], [pressure1.-(sum₁/len₁₂), pressure2.-(sum₂/len₁₂)],
# 		labels=permutedims(["pressure on the bottom wall", "pressure on the top wall"]),
# 		xlabel="scaled distance",
# 		ylabel="scaled pressure",
# 		ylims=(-2,2))
# 	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
# end

# CSV.write("C:/Users/blagn771/Desktop/TopPressure.csv", Tables.table(pressureTop), writeheader=false)
# CSV.write("C:/Users/blagn771/Desktop/BottomPressure.csv", Tables.table(pressureBottom), writeheader=false)