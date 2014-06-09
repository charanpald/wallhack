
import sympy 

"""
Check our calculations are correct 
"""

ui, vp, vq, ri, rho = sympy.symbols('ui vp vq ri rho')
print(sympy.diff((1-ui*vp+ui*vq)**2 * (1-rho*ui*vp+rho*ri), ui))

