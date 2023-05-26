using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

fit = y -> scale(
    interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
    range(0,1,length=length(y))
)

width = [0.02, 0.06, 0.06, 0.05, 0.03, 0.015, 0.01]
thk = fit(width)

function fish(thk; L=2^6, Re=1e4, n, m)
    # fraction along the fish length
    s(x) = clamp(x[1]/L, 0, 1)

    # fish geometry: thickened line SDF
    sdf(x,t) = √sum(abs2, x - L*SVector(s(x), 0.)) - L*thk(s(x))

    # fish motion: position offset and travelling path
    distInit = 51
    wallThk = 29
    function map(x,t)
        return x - [n-L-distInit,m-wallThk-20.5*4] + [4.4*t,0.] #m-wallThk-20.5
    end

    # make the simulation
	return AutoBody(sdf,map)
end