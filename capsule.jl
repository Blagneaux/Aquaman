using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images, Statistics
using LinearAlgebra: norm2

fit = y -> scale(
        interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
        range(0,1,length=length(y))
    )

width = [0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]
thk = fit(width)

@fastmath kernᵦ(d) = 0.5+0.5d+0.5sin(π*d)/π
μᵦ(d,ϵ) = kernᵦ(clamp(d/ϵ,-1,1))

function capsule(L=71.2-6.5, St=0.61, A=0.466, U=0.89;n,m)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdf(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))

    # fish motion: travelling wave
    pivot = 9.7
    distInit = 51
    f = St * U/(2A * L)
    print("pulse: ",2π*f,"\n")
    
    function map(x,t)
        while t <= 4/f
            xc = x - [n-L-distInit + pivot,m/2] + [t,0.]
            amp = 12.5π/180
            α = amp*sin(2π*f*t)
            R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
            I₂ = @SMatrix [1 0; 0 1]
            xc = (μᵦ(xc[1],10).*R + (1-μᵦ(xc[1],10)).*I₂)*xc
            return xc + [pivot,0.]
        end
        xc = x - [n-L-distInit + pivot,m/2]+ [4/f,0.]
        α = 0
        R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        I₂ = @SMatrix [1 0; 0 1]
        xc = (μᵦ(xc[1],10).*R + (1-μᵦ(xc[1],10)).*I₂)*xc
        return xc + [pivot,0.]
    end

	# make the simulation
	return AutoBody(sdf,map)
    # return Simulation((642,258), [0.,0.], (L+6.5), U=0.89; ν=U*(L+6.5)/6070, body=AutoBody(sdf, map))
end

# L,A,St,U = 71.2-6.5,0.466,0.61,0.89
# swimmer = capsule()
# period = 2A/St
# cycle = range(0, period*8*23/24, length=24*8)

# # plot the pressure scaled by the body length L and flow speed U
# function plot_pressure(sim, t)
# 	contourf(sim.flow.p', connectgaps=false,
# 			 clims=(-2, 2), legend=true, border=:none)
#     print("Mean: ",mean(sim.flow.p),"\n")
# end


# # make a gif over a swimming cycle
# @gif for t ∈ sim_time(swimmer) .+ cycle
# 	sim_step!(swimmer, t, remeasure=true, verbose=false)
# 	plot_pressure(swimmer, t)
# end

# Final and max Cp = 464241
# Put back into real pressure, this roughtly makes 37bars.
# With real life experiments, the pressure we get is around 2Pa on the top and bottom edges,
# which is also around the same order of magnitude I get when substracting the mean to the whole simulation.