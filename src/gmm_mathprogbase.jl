# ----------------------------- #
# MathProgBase solver interface #
# ----------------------------- #

function SolverInterface.initialize(d::GMMNLPE, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end


SolverInterface.features_available(d::GMMNLPE) = [:Grad, :Jac, :Hess]

function SolverInterface.eval_f(d::GMMNLPE, theta)
    gn = d.smf(theta)
    (gn'd.W*gn)[1]
end

SolverInterface.eval_g(d::GMMNLPE, Dg, x) = nothing

function SolverInterface.eval_grad_f(d::GMMNLPE, grad_f, theta)
    grad_f[:] = 2*d.Dmf(theta)'*(d.W*d.smf(theta))
end

SolverInterface.jac_structure(d::GMMNLPE) = [],[]
SolverInterface.eval_jac_g(d::GMMNLPE, J, x) = nothing
SolverInterface.eval_hesslag(d::GMMNLPE, H, x, σ, μ) = nothing
SolverInterface.hesslag_structure(d::GMMNLPE) = [],[]
