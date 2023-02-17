using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

function block(L=2^5;Re=250,U=1,amp=π/4,ϵ=0.5,thk=2ϵ+√2)
    # Line segment SDF
    function sdf(x,t)
        y = x .- SVector(0.,clamp(x[2],-L/2,L/2))
        √sum(abs2,y)-thk/2
    end
    # Oscillating motion and rotation
    function map(x,t)
        α = amp*cos(t*U/L); R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        R * (x.-SVector(3L-L*sin(t*U/L),4L))
    end
    Simulation((6L+2,6L+2),zeros(2),L;U,ν=U*L/Re,body=AutoBody(sdf,map),ϵ)
end

swimmer = block()
cycle = range(0, 4π, length=24*8)

@gif for t ∈ cycle
	measure!(swimmer, t*swimmer.L/swimmer.U)
	contour(swimmer.flow.μ₀[:,:,1]',
			aspect_ratio=:equal, legend=true, border=:none)
end

# plot the pressure scaled by the body length L and flow speed U
function plot_pressure(sim, t)
	contourf(sim.flow.p',
			 clims=(-1000, 1000), linewidth=0.1,
			 aspect_ratio=:equal, legend=true, border=:none)
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	plot_pressure(swimmer, t)
end