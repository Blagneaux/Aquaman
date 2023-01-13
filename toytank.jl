using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function circle(radius=8;Re=250,n=10,m=6)
    center, ν = radius*m/2, radius/Re
    R = [4,2]
    sdf(x,t) = √(max(x[1]-R[1],0)^2 + max(x[2]-R[2],0)^2)- n*radius/2
    function map(x,t)
        return x - [n*radius/2,m*radius/2]
    end
    Simulation((n*radius+2,m*radius+2), [1.,0.], radius; ν, body=AutoBody(sdf))
end

function computeSDF(sim, t)
    s = copy(sim.flow.p)
    for I ∈ inside(s)
        x = loc(0, I)
        s[I] = sim.body.sdf(x,t*circle(20).L/circle(20).U)::Float64
    end
    contourf(s', clims=(-10,20), linewidth=0,
            aspect_ratio=:equal, legend=true, border=:none)
end

@gif for t ∈ sim_time(circle(20))
    sim_step!(circle(20), t, remeasure=true, verbose=true)
    computeSDF(circle(20), t)
end