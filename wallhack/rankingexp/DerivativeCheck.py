
import sympy 

"""
Check our calculations are correct 
"""

ui, vp, vq, ri, rho = sympy.symbols('ui vp vq ri rho')

print("\nObjective 1")
print("-"*10)
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri)**2, ui))
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri)**2, vp))
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri)**2, vq))

print("\nObjective 2")
print("-"*10)
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri)**0.5, ui))
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri)**0.5, vp))
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri)**0.5, vq))

print("\nObjective 3")
print("-"*10)
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * sympy.tanh(1-rho*ui*vp+rho*ri), ui))
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * sympy.tanh(1-rho*ui*vp+rho*ri), vp))
sympy.pprint(sympy.diff((1-ui*vp+ui*vq)**2 * sympy.tanh(1-rho*ui*vp+rho*ri), vq))
