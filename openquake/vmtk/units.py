import numpy as np

class units():
    
    """
    Standardised units in terms of m and kn for Opensees applications
    """
    pi = np.pi
    g = 9.81
    
    # Length
    m = 1.0
    mm = m/1000.
    cm = m/100.
    inch = 25.4*mm
    ft = 12*inch
    
    # Area
    m2 = np.power(m, 2)
    mm2 = np.power(mm, 2)
    cm2 = np.power(cm, 2)
    inch2 = np.power(inch, 2)
    
    # Second Moment of Area
    m4 = np.power(m, 4)
    cm4 = np.power(cm, 4)
    mm4 = np.power(mm, 4)
    inch4 = np.power(inch, 4)
    
    # Force
    kN = 1.0
    N = kN/1000.
    kips = kN*4.448221615
    
    # Moment
    kNm = kN*m
    
    # Mass (tonnes)
    tonne = 1.0;
    kg = tonne/1000.
    
    # Stress (kN/m2 or kPa)
    kNm2 = kN/m2
    Pa = N/m2
    kPa = Pa*1.0e3
    MPa = Pa*1.0e6
    Nmm2 = N/mm2
    kNmm2 = Nmm2*1.0e3
    GPa = Pa*1.0e9
    ksi = 6.8947573*MPa
    kgcm2 = kg*g/cm2
    
    # Angles
    degrees = pi/180.

    # Unit weights
    kNm3 = kN/m/m/m
