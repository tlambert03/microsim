# Excitation and Emission Methodology
This document outlines the mechanism of how the excitation and emission of the fluoropore is simulated in this package. The code is present at [emission.py](../src/microsim/schema/_emission.py). 

## Excitation
  The task is to estimate the number of photons absorbed by the fluorophores for each `1nm` wavelength interval. For this, following steps are followed:

1. Compute the irradiance for every `1nm` wavelength interval of the light source. `irradiance = ex_filter_spectrum * light_power`.  
2. Calculate the absorption cross section given the extinction coefficient. See [section below](#absorption-cross-section) for more details. This is inturn used to calculate the absorbtion cross section for every `1nm` wavelength interval of the fluorophore excitation spectrum.
3. Now that we have the abrobption cross section and irradiance, we calculate the power absorbed by the fluorophores for every `1nm` wavelength interval.
4. Finally, we compute the absorbed photon count by simply dividing the power absorbed by the energy of a single photon.

## Emission 
  For emission, we need to estimate the emission flux for every `1nm` wavelength interval which is done as follows:
1. We scale up the emission spectrum by a factor to account for light power, extinction coefficient and overlap between fluorophore excitation spectrum and input light spectrum. `scaling_factor = ex_rate.integral() / fluor_em_spectrum.integral()`
2. Multiply the scaled emission spectrum by the quantum yield to get the number of emission events. `fluor_em_rate = fluor_em_rate * fluor.quantum_yield`
3. Apply the emission filter(s) to the emission spectrum.

 
### Absorption Cross Section
Given the extinction coefficient, we calculate the absorption cross section using the following formula:

> cross section = 1000 * ec * np.log(10) / $N_{A}$,

where $N_{A}$ is the Avogadro number. The factor `1000` appears because we convert the concentration from `mol/L` to `mol/cm3`. Note that this conversion is taken care by [pint](https://pint.readthedocs.io/en/stable/) and therefore is not explicit in the code. The natural logarithm is converted to base 10 logarithm by multiplying with `np.log(10)`. [Reference 1](https://en.wikipedia.org/wiki/Absorption_cross_section), [Reference 2](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Time_Dependent_Quantum_Mechanics_and_Spectroscopy_%28Tokmakoff%29/07:_Interaction_of_Light_and_Matter/7.05:_Absorption_Cross-Sections)


