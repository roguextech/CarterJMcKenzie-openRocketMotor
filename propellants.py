class QDL:
    density = 1796.426  # kg/m
    a = 0.000027099  # m/(s*pa^n)
    n = 0.35
    T = 3552.3
    gamma = 1.164
    molar_mass = 26.935
    c_star = 1780.53
    reference_pressure = 6894800
    reference_burn_rate = 0.0067056
    R = 8314 / molar_mass


class ANB3066:
    density = 1765.978  # kg/m
    a = 0.0001392  # m/(s*pa^n)
    n = 0.27
    T = 3497.6
    gamma = 1.1285
    molar_mass = 26.933
    c_star = 1809.6
    reference_pressure = 6894800  # (1000 psi)
    reference_burn_rate = 0.009779
    R = 8314 / molar_mass


class QDT:
    density = 1799.19
    a = 0.000071956
    n = 0.3
    T = 3533
    gamma = 1.167
    molar_mass = 30.51
    c_star = 1772.147
    reference_pressure = 6894800  # (1000 psi)
    reference_burn_rate = 0.0081026
    R = 8314 / molar_mass


class UTP3001:
    density = 1760.4419
    a = 0.00019332
    n = 0.25
    T = 3409
    gamma = 1.3373
    molar_mass = 26.4206
    c_star = 1563.46
    reference_pressure = 6894800.0
    reference_burn_rate = 0.009906
    R = 8314 / molar_mass


class TPH3062:
    density = 1738.298
    a = 0.000067605
    n = 0.3
    T = 3201.5
    gamma = 1.192
    molar_mass = 26.06
    c_star = 1659.3569
    reference_pressure = 6894800.0
    reference_burn_rate = 0.00762
    R = 8314 / molar_mass


class TPH1148:
    density = 1757.674
    a = 0.000045165
    n = 0.35
    T = 3425
    gamma = 1.1828
    molar_mass = 29.323
    c_star = 1726.698
    reference_pressure = 6894800.0
    reference_burn_rate = 0.011176
    R = 8314 / molar_mass


class CherryLimeade:
    density = 1679.99
    a = 0.00003517
    n = 0.3273  # nondim
    T = 3500  # Kelvin
    molar_mass = 23.67  # g/mol
    gamma = 1.21  # nondim
    c_star = 1703.8  # ft/s (5624 by my formula)
    R = 8314 / molar_mass


class BlueThunder:
    density = 1625.07
    a = 0.00006995
    n = 0.321000
    T = 2616.5  # Kelvin
    molar_mass = 22.959  # g/mol
    gamma = 1.235  # nondim
    c_star = 1471  # ft/s ( by my formula)
    R = 8314 / molar_mass


class NakkaKNDX:
    density = 1784.97
    a = 0.000010538
    n = 0.444  # nondim
    T = 1625  # Kelvin
    molar_mass = 42.39  # g/mol
    gamma = 1.1308   # nondim
    c_star = 889.102  # ft/s ( by my formula)
    R = 8314 / molar_mass
