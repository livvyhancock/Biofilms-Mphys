# biofilms
The code was developed to fit equivalent circuit models to
data, outputting the parameter values of elements in the equivalent
circuit that optimised the fit. The code is designed
for EIS data, working with the real and imaginary values of
impedance at each frequency. Firstly a function is defined for
the desired equivalent circuit models. The function takes frequency
and parameters used in the equivalent circuit model
as arguments. It returns real and imaginary impedance based
on the equation specific to the circuit. A fitting procedure then
compares data produced by the model to the experimental data
and optimises the model parameters. See also the PDF report for summary of findings and methodology.
